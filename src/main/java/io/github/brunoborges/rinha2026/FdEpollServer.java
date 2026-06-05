package io.github.brunoborges.rinha2026;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

import static io.github.brunoborges.rinha2026.LinuxSyscalls.*;

/**
 * Single-thread Linux event loop that replaces the JDK {@code com.sun.net.httpserver} server.
 *
 * <p>The load balancer (lapada) accepts client TCP connections on :9999 and hands each accepted socket
 * fd to this process via {@code sendmsg(SCM_RIGHTS)} over a persistent Unix control socket. This loop:
 * <ol>
 *   <li>binds/listens the control socket ({@code FD_SOCKET});</li>
 *   <li>{@code recvmsg(SCM_RIGHTS)} each client fd and adds it to {@code epoll};</li>
 *   <li>reads, parses, scores, and writes the response directly on the client fd via FFM
 *       {@code recv}/{@code send} — the LB is never on the data path.</li>
 * </ol>
 *
 * <p>Everything runs on one platform thread (one CPU budget), with no {@code sun.nio.ch} reflection and
 * no per-request heap allocation in steady state (pre-framed off-heap responses, a reusable off-heap
 * receive buffer, and a per-connection reusable parse buffer). Linux x86-64 only.
 */
final class FdEpollServer {

    private static final Logger LOG = Logger.getLogger(FdEpollServer.class.getName());

    private static final int MSG_NOSIGNAL = 0x4000;
    private static final int MSG_CMSG_CLOEXEC = 0x40000000;
    private static final int MAX_EVENTS = 256;
    private static final int FD_TABLE_SIZE = 1 << 16;

    // Per-fd dispatch markers stored in the fd table.
    private static final Object LISTEN_MARK = new Object();
    private static final Object CONTROL_MARK = new Object();

    // pump() result codes.
    private static final int S_SENT = 0;
    private static final int S_BLOCKED = 1;
    private static final int S_ERROR = 2;

    private final String controlSocketPath;
    private final HttpRouter router;

    private final Object[] fdState = new Object[FD_TABLE_SIZE];
    private final int[] generation = new int[FD_TABLE_SIZE];

    // Native scratch (confined to the loop thread, allocated in run()).
    private Arena arena;
    private MemorySegment cap;       // errno capture
    private MemorySegment recvScratch; // off-heap receive buffer
    private MemorySegment events;    // epoll_event[MAX_EVENTS]
    private MemorySegment evScratch;  // one epoll_event for epoll_ctl
    // recvmsg(SCM_RIGHTS) buffers
    private MemorySegment msghdr;
    private MemorySegment iovec;
    private MemorySegment dataByte;
    private MemorySegment cmsgBuf;

    private int epfd = -1;

    FdEpollServer(String controlSocketPath, HttpRouter router) {
        this.controlSocketPath = controlSocketPath;
        this.router = router;
    }

    /** Per-client connection state. */
    private static final class Conn {
        final HttpConnection http = new HttpConnection();
        int pendingIdx = -1;   // response currently being written, or -1
        int pendingOff = 0;
        boolean advanceAfterDrain = false; // advance the parse buffer once the pending write drains
        boolean closeAfterWrite = false;
        boolean outArmed = false;
    }

    /** Start the loop on a dedicated non-daemon platform thread; blocks until the control socket is bound. */
    void start() {
        java.util.concurrent.CountDownLatch bound = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.atomic.AtomicReference<Throwable> err = new java.util.concurrent.atomic.AtomicReference<>();
        Thread t = new Thread(() -> run(bound, err), "rinha-epoll");
        t.setDaemon(false);
        t.setPriority(Thread.MAX_PRIORITY);
        t.start();
        try {
            bound.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for epoll bind", e);
        }
        Throwable failure = err.get();
        if (failure != null) {
            throw new IllegalStateException("epoll server failed to start", failure);
        }
    }

    private void run(java.util.concurrent.CountDownLatch bound,
                     java.util.concurrent.atomic.AtomicReference<Throwable> err) {
        try (Arena a = Arena.ofConfined()) {
            this.arena = a;
            this.cap = newCapture(a);
            this.recvScratch = a.allocate(HttpConnection.BUF_SIZE);
            this.events = a.allocate((long) EPOLL_EVENT_SIZE * MAX_EVENTS, 8);
            this.evScratch = a.allocate(EPOLL_EVENT_SIZE, 8);
            initRecvmsgBuffers(a);

            int listenFd = bindControlSocket();
            this.epfd = epollCreate1(cap, EPOLL_CLOEXEC);
            if (epfd < 0) {
                throw new IllegalStateException("epoll_create1 failed errno=" + errnoOf(cap));
            }
            fdState[listenFd] = LISTEN_MARK;
            epollAdd(listenFd, EPOLLIN);
            LOG.info(() -> "epoll loop serving; control socket " + controlSocketPath);

            bound.countDown(); // socket is bound + registered: safe to publish readiness
            loop();
        } catch (Throwable t) {
            err.set(t);
            bound.countDown();
            LOG.severe("epoll loop terminated: " + t);
        }
    }

    private void loop() throws Throwable {
        while (true) {
            int n = epollWait(cap, epfd, events, MAX_EVENTS, -1);
            if (n < 0) {
                int e = errnoOf(cap);
                if (e == EINTR) {
                    continue;
                }
                throw new IllegalStateException("epoll_wait failed errno=" + e);
            }
            for (int k = 0; k < n; k++) {
                long base = (long) k * EPOLL_EVENT_SIZE;
                int mask = events.get(ValueLayout.JAVA_INT, base + EPOLL_EVENT_OFF_EVENTS);
                long token = events.get(ValueLayout.JAVA_LONG_UNALIGNED, base + EPOLL_EVENT_OFF_DATA);
                int fd = (int) (token & 0xFFFFFFFFL);
                int gen = (int) (token >>> 32);
                if (gen != generation[fd]) {
                    continue; // stale event for an fd that was closed (and possibly reused) this batch
                }
                Object st = fdState[fd];
                if (st == LISTEN_MARK) {
                    acceptControlConnections(fd);
                } else if (st == CONTROL_MARK) {
                    drainControl(fd);
                } else if (st instanceof Conn conn) {
                    // Drain readable bytes first: a peer that closed right after writing delivers
                    // EPOLLHUP together with EPOLLIN, and the buffered request must still be served.
                    if ((mask & (EPOLLIN | EPOLLRDHUP)) != 0) {
                        onClientReadable(fd, conn);
                        if (fdState[fd] != conn) {
                            continue; // closed during read
                        }
                    }
                    if ((mask & EPOLLOUT) != 0) {
                        onClientWritable(fd, conn);
                        if (fdState[fd] != conn) {
                            continue;
                        }
                    }
                    if ((mask & (EPOLLHUP | EPOLLERR)) != 0) {
                        closeClient(fd);
                    }
                }
            }
        }
    }

    // ---- control socket -----------------------------------------------------

    private void acceptControlConnections(int listenFd) throws Throwable {
        while (true) {
            int cfd = accept4(cap, listenFd, SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd >= 0) {
                if (cfd >= FD_TABLE_SIZE) {
                    closeFd(cfd);
                    continue;
                }
                fdState[cfd] = CONTROL_MARK;
                epollAdd(cfd, EPOLLIN);
                continue;
            }
            int e = errnoOf(cap);
            if (e == EINTR) {
                continue;
            }
            if (e == EAGAIN || e == EWOULDBLOCK) {
                return;
            }
            return; // transient accept error; the listener stays registered
        }
    }

    /** Receive as many passed client fds as are queued on this control connection. */
    private void drainControl(int controlFd) throws Throwable {
        while (true) {
            int clientFd = receiveFd(controlFd);
            if (clientFd == FD_EAGAIN) {
                return;
            }
            if (clientFd < 0) {
                // Control connection closed by lapada (restart) or fatal error.
                epollDel(controlFd);
                fdState[controlFd] = null;
                generation[controlFd]++;
                closeFd(controlFd);
                return;
            }
            if (clientFd >= FD_TABLE_SIZE) {
                closeFd(clientFd);
                continue;
            }
            Conn conn = new Conn();
            fdState[clientFd] = conn;
            epollAdd(clientFd, EPOLLIN | EPOLLRDHUP);
        }
    }

    private static final int FD_EAGAIN = Integer.MIN_VALUE;

    /**
     * {@code recvmsg(SCM_RIGHTS)} one fd from {@code controlFd}. Returns the fd (&ge;0), {@link #FD_EAGAIN}
     * when no message is queued, or -1 on close / malformed ancillary data.
     */
    private int receiveFd(int controlFd) throws Throwable {
        dataByte.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);
        cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_DATA, -1);
        msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN, CMSG_SPACE);

        long r;
        while (true) {
            r = recvmsg(cap, controlFd, msghdr, MSG_CMSG_CLOEXEC);
            if (r >= 0) {
                break;
            }
            int e = errnoOf(cap);
            if (e == EINTR) {
                continue;
            }
            if (e == EAGAIN || e == EWOULDBLOCK) {
                return FD_EAGAIN;
            }
            return -1;
        }
        if (r == 0) {
            return -1; // peer closed
        }
        long ctrlLen = msghdr.get(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN);
        if (ctrlLen < CMSG_LEN_VALUE) {
            return -1;
        }
        long cmsgLen = cmsgBuf.get(ValueLayout.JAVA_LONG, OFF_CMSG_LEN);
        int level = cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_LEVEL);
        int type = cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_TYPE);
        if (cmsgLen < CMSG_LEN_VALUE || level != SOL_SOCKET || type != SCM_RIGHTS) {
            return -1;
        }
        return cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_DATA);
    }

    // ---- client I/O ---------------------------------------------------------

    private void onClientReadable(int fd, Conn conn) throws Throwable {
        if (conn.pendingIdx >= 0) {
            return; // still flushing a response; resume reading after EPOLLOUT drains it
        }
        while (true) {
            int space = HttpConnection.BUF_SIZE - conn.http.pos;
            if (space == 0) {
                // Request did not complete within the buffer: reject and close to avoid stream desync.
                conn.closeAfterWrite = true;
                int st = beginSend(fd, conn, HttpResponses.RESP_PAYLOAD_TOO_LARGE);
                if (st != S_BLOCKED) {
                    closeClient(fd); // fully sent or errored: tear down now
                }
                return;
            }
            long nr = recv(cap, fd, recvScratch, space, 0);
            if (nr > 0) {
                MemorySegment.copy(recvScratch, ValueLayout.JAVA_BYTE, 0L,
                        conn.http.buf, conn.http.pos, (int) nr);
                conn.http.pos += (int) nr;
                int status = processBuffered(fd, conn);
                if (status != S_SENT) {
                    return; // closed or blocked on write
                }
                if (nr < space) {
                    return; // socket drained for now; next EPOLLIN resumes
                }
                // buffer was filled; keep reading
            } else if (nr == 0) {
                closeClient(fd); // orderly peer close
                return;
            } else {
                int e = errnoOf(cap);
                if (e == EINTR) {
                    continue;
                }
                if (e == EAGAIN || e == EWOULDBLOCK) {
                    return;
                }
                closeClient(fd);
                return;
            }
        }
    }

    /**
     * Parse and serve every complete request currently buffered. Returns {@link #S_SENT} when the buffer
     * is fully processed (caller may read more), {@link #S_BLOCKED} when a write blocked (EPOLLOUT armed),
     * or a closed state when the connection was torn down.
     */
    private int processBuffered(int fd, Conn conn) throws Throwable {
        while (true) {
            int r = conn.http.tryParse();
            if (r == HttpConnection.NEED_MORE) {
                return S_SENT;
            }
            if (r == HttpConnection.TOO_LARGE) {
                conn.closeAfterWrite = true;
                int st = beginSend(fd, conn, HttpResponses.RESP_PAYLOAD_TOO_LARGE);
                if (st != S_BLOCKED) {
                    closeClient(fd); // fully sent or errored
                    return S_ERROR;
                }
                return S_BLOCKED; // armed; closeAfterWrite tears it down once drained
            }
            int idx = router.responseIndex(conn.http);
            int sent = beginSend(fd, conn, idx);
            if (sent == S_BLOCKED) {
                conn.advanceAfterDrain = true;
                return S_BLOCKED;
            }
            if (sent == S_ERROR) {
                closeClient(fd);
                return S_ERROR;
            }
            conn.http.advanceAfterRequest();
            if (conn.closeAfterWrite) {
                closeClient(fd);
                return S_ERROR;
            }
        }
    }

    private void onClientWritable(int fd, Conn conn) throws Throwable {
        if (conn.pendingIdx < 0) {
            if (!disarmOut(fd, conn)) {
                closeClient(fd);
            }
            return;
        }
        int st = pump(fd, conn);
        if (st == S_BLOCKED) {
            return; // stay armed for EPOLLOUT
        }
        if (st == S_ERROR) {
            closeClient(fd);
            return;
        }
        conn.pendingIdx = -1;
        if (conn.advanceAfterDrain) {
            conn.http.advanceAfterRequest();
            conn.advanceAfterDrain = false;
        }
        if (conn.closeAfterWrite) {
            closeClient(fd);
            return;
        }
        if (!disarmOut(fd, conn)) {
            closeClient(fd);
            return;
        }
        // Serve any pipelined requests that arrived while we were blocked.
        processBuffered(fd, conn);
    }

    /** Begin writing response {@code idx}; returns {@link #S_SENT}, {@link #S_BLOCKED}, or {@link #S_ERROR}. */
    private int beginSend(int fd, Conn conn, int idx) throws Throwable {
        conn.pendingIdx = idx;
        conn.pendingOff = 0;
        int st = pump(fd, conn);
        if (st == S_SENT) {
            conn.pendingIdx = -1;
        } else if (st == S_BLOCKED) {
            if (!armOut(fd, conn)) {
                return S_ERROR;
            }
        }
        return st;
    }

    /** Drive {@code send} for the in-flight response from {@code pendingOff} to completion or EAGAIN. */
    private int pump(int fd, Conn conn) throws Throwable {
        int idx = conn.pendingIdx;
        int len = HttpResponses.length(idx);
        MemorySegment seg = HttpResponses.segment(idx);
        int off = conn.pendingOff;
        while (off < len) {
            long s = send(cap, fd, seg.asSlice(off, len - off), len - off, MSG_NOSIGNAL);
            if (s > 0) {
                off += (int) s;
            } else {
                int e = errnoOf(cap);
                if (e == EINTR) {
                    continue;
                }
                if (e == EAGAIN || e == EWOULDBLOCK) {
                    conn.pendingOff = off;
                    return S_BLOCKED;
                }
                return S_ERROR;
            }
        }
        conn.pendingOff = len;
        return S_SENT;
    }

    // ---- epoll / fd helpers -------------------------------------------------

    /**
     * Arm EPOLLOUT and drop read interest while a response is in flight. Suppressing EPOLLIN avoids a
     * level-triggered busy-loop when the client has also pipelined more bytes we cannot yet process.
     * EPOLLHUP/EPOLLERR are always reported regardless of the mask, so hangups are still detected.
     */
    private boolean armOut(int fd, Conn conn) {
        if (!conn.outArmed) {
            if (!epollMod(fd, EPOLLOUT)) {
                return false;
            }
            conn.outArmed = true;
        }
        return true;
    }

    private boolean disarmOut(int fd, Conn conn) {
        if (conn.outArmed) {
            if (!epollMod(fd, EPOLLIN | EPOLLRDHUP)) {
                return false;
            }
            conn.outArmed = false;
        }
        return true;
    }

    private void closeClient(int fd) {
        try {
            epollDel(fd);
        } catch (Throwable ignored) {
            // fd may already be gone
        }
        fdState[fd] = null;
        generation[fd]++; // invalidate any stale events already returned for this fd this batch
        closeFd(fd);
    }

    private void closeFd(int fd) {
        try {
            close(cap, fd);
        } catch (Throwable ignored) {
        }
    }

    private void epollAdd(int fd, int mask) throws Throwable {
        writeEvent(mask, fd);
        if (epollCtl(cap, epfd, EPOLL_CTL_ADD, fd, evScratch) < 0) {
            throw new IllegalStateException("epoll_ctl ADD fd=" + fd + " errno=" + errnoOf(cap));
        }
    }

    /** Returns {@code true} on success; a failed MOD means the caller should drop the connection. */
    private boolean epollMod(int fd, int mask) {
        writeEvent(mask, fd);
        try {
            return epollCtl(cap, epfd, EPOLL_CTL_MOD, fd, evScratch) >= 0;
        } catch (Throwable t) {
            return false;
        }
    }

    private void epollDel(int fd) throws Throwable {
        epollCtl(cap, epfd, EPOLL_CTL_DEL, fd, MemorySegment.NULL);
    }

    private void writeEvent(int mask, int fd) {
        long token = ((long) generation[fd] << 32) | (fd & 0xFFFFFFFFL);
        evScratch.set(ValueLayout.JAVA_INT, EPOLL_EVENT_OFF_EVENTS, mask);
        evScratch.set(ValueLayout.JAVA_LONG_UNALIGNED, EPOLL_EVENT_OFF_DATA, token);
    }

    // ---- setup --------------------------------------------------------------

    private void initRecvmsgBuffers(Arena a) {
        msghdr = a.allocate(MSGHDR_SIZE, 8);
        iovec = a.allocate(IOVEC_SIZE, 8);
        dataByte = a.allocate(1);
        cmsgBuf = a.allocate(CMSG_SPACE, 8);

        iovec.set(ValueLayout.ADDRESS, OFF_IOV_BASE, dataByte);
        iovec.set(ValueLayout.JAVA_LONG, OFF_IOV_LEN, 1L);

        msghdr.set(ValueLayout.ADDRESS, OFF_MSG_IOV, iovec);
        msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_IOVLEN, 1L);
        msghdr.set(ValueLayout.ADDRESS, OFF_MSG_CONTROL, cmsgBuf);
        msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN, CMSG_SPACE);

        cmsgBuf.set(ValueLayout.JAVA_LONG, OFF_CMSG_LEN, CMSG_LEN_VALUE);
        cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_LEVEL, SOL_SOCKET);
        cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_TYPE, SCM_RIGHTS);
    }

    private int bindControlSocket() throws Throwable {
        byte[] pathBytes = controlSocketPath.getBytes(StandardCharsets.UTF_8);
        if (pathBytes.length >= SOCKADDR_UN_SIZE - OFF_SUN_PATH) {
            throw new IllegalArgumentException("control socket path too long: " + controlSocketPath);
        }
        // Remove any stale socket inode from a previous run.
        MemorySegment cPath = arena.allocateFrom(controlSocketPath);
        unlink(cap, cPath);

        int fd = socket(cap, AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) {
            throw new IllegalStateException("socket(AF_UNIX) failed errno=" + errnoOf(cap));
        }
        MemorySegment addr = arena.allocate(SOCKADDR_UN_SIZE);
        addr.set(ValueLayout.JAVA_SHORT, OFF_SUN_FAMILY, (short) AF_UNIX);
        MemorySegment.copy(pathBytes, 0, addr, ValueLayout.JAVA_BYTE, OFF_SUN_PATH, pathBytes.length);
        int addrLen = OFF_SUN_PATH + pathBytes.length + 1; // include NUL terminator
        if (bind(cap, fd, addr, addrLen) < 0) {
            throw new IllegalStateException("bind(" + controlSocketPath + ") failed errno=" + errnoOf(cap));
        }
        if (listen(cap, fd, 1024) < 0) {
            throw new IllegalStateException("listen failed errno=" + errnoOf(cap));
        }
        // Make the socket connectable by the load balancer even if it runs under a different UID.
        if (chmod(cap, cPath, 0666) < 0) {
            LOG.warning(() -> "chmod 0666 on " + controlSocketPath + " failed errno=" + errnoOf(cap));
        }
        return fd;
    }
}
