package io.github.brunoborges.rinha2026;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;

/**
 * Thin Panama FFM bindings to the Linux x86-64 syscalls the event loop needs. The same FFM mechanism
 * already backs the off-heap dataset {@code mmap}, so these downcalls run under GraalVM native image
 * without any reflection metadata.
 *
 * <p>Every handle captures {@code errno} (prepended as the first {@link MemorySegment} argument) so the
 * non-blocking loop can distinguish {@code EAGAIN}/{@code EINTR} from fatal errors. All constants and
 * struct offsets are Linux x86-64 / glibc; this class is intentionally not portable (see repo target).
 *
 * <p>Class initialization resolves the libc symbols eagerly and therefore must only happen on Linux
 * (i.e. at server start on the contest box), never from host unit tests.
 */
final class LinuxSyscalls {

    // ---- socket / address-family constants ----
    static final int AF_UNIX = 1;
    static final int SOCK_STREAM = 1;
    static final int SOCK_NONBLOCK = 0x800;   // octal 04000
    static final int SOCK_CLOEXEC = 0x80000;  // octal 02000000

    // ---- fcntl ----
    static final int F_GETFL = 3;
    static final int F_SETFL = 4;
    static final int O_NONBLOCK = 0x800;

    // ---- errno ----
    static final int EAGAIN = 11;
    static final int EWOULDBLOCK = 11;
    static final int EINTR = 4;

    // ---- epoll ----
    static final int EPOLL_CLOEXEC = 0x80000;
    static final int EPOLL_CTL_ADD = 1;
    static final int EPOLL_CTL_DEL = 2;
    static final int EPOLL_CTL_MOD = 3;
    static final int EPOLLIN = 0x001;
    static final int EPOLLOUT = 0x004;
    static final int EPOLLERR = 0x008;
    static final int EPOLLHUP = 0x010;
    static final int EPOLLRDHUP = 0x2000;

    // struct epoll_event is __packed__ on x86-64: { u32 events; u64 data; } => size 12, data at offset 4.
    static final int EPOLL_EVENT_SIZE = 12;
    static final int EPOLL_EVENT_OFF_EVENTS = 0;
    static final int EPOLL_EVENT_OFF_DATA = 4;

    // ---- SCM_RIGHTS (one fd) — see jvmoonshot FdPassing; glibc 64-bit layout ----
    static final int SOL_SOCKET = 1;
    static final int SCM_RIGHTS = 1;
    static final int TCP_NODELAY = 1;
    static final int IPPROTO_TCP = 6;

    static final long MSGHDR_SIZE = 56;
    static final long OFF_MSG_IOV = 16;
    static final long OFF_MSG_IOVLEN = 24;
    static final long OFF_MSG_CONTROL = 32;
    static final long OFF_MSG_CONTROLLEN = 40;
    static final long IOVEC_SIZE = 16;
    static final long OFF_IOV_BASE = 0;
    static final long OFF_IOV_LEN = 8;
    static final long CMSG_SPACE = 24;        // CMSG_SPACE(sizeof(int))
    static final long CMSG_LEN_VALUE = 20;    // CMSG_LEN(sizeof(int)) = 16 + 4
    static final long OFF_CMSG_LEN = 0;
    static final long OFF_CMSG_LEVEL = 8;
    static final long OFF_CMSG_TYPE = 12;
    static final long OFF_CMSG_DATA = 16;

    // struct sockaddr_un { u16 sun_family; char sun_path[108]; }
    static final int SOCKADDR_UN_SIZE = 110;
    static final int OFF_SUN_FAMILY = 0;
    static final int OFF_SUN_PATH = 2;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBC = LINKER.defaultLookup();

    private static final MemoryLayout CAPTURE = Linker.Option.captureStateLayout();
    private static final VarHandle ERRNO_VH =
            CAPTURE.varHandle(MemoryLayout.PathElement.groupElement("errno"));

    private static final MethodHandle SOCKET;
    private static final MethodHandle BIND;
    private static final MethodHandle LISTEN;
    private static final MethodHandle ACCEPT4;
    private static final MethodHandle RECV;
    private static final MethodHandle SEND;
    private static final MethodHandle RECVMSG;
    private static final MethodHandle CLOSE;
    private static final MethodHandle FCNTL;
    private static final MethodHandle SETSOCKOPT;
    private static final MethodHandle EPOLL_CREATE1;
    private static final MethodHandle EPOLL_CTL;
    private static final MethodHandle EPOLL_WAIT;
    private static final MethodHandle UNLINK;
    private static final MethodHandle CHMOD;

    static {
        SOCKET = dc("socket");
        BIND = dc("bind");
        LISTEN = dc("listen");
        ACCEPT4 = dc("accept4");
        RECV = dc("recv");
        SEND = dc("send");
        RECVMSG = dc("recvmsg");
        CLOSE = dc("close");
        FCNTL = dc("fcntl");
        SETSOCKOPT = dc("setsockopt");
        EPOLL_CREATE1 = dc("epoll_create1");
        EPOLL_CTL = dc("epoll_ctl");
        EPOLL_WAIT = dc("epoll_wait");
        UNLINK = dc("unlink");
        CHMOD = dc("chmod");
    }

    private static MethodHandle dc(String name) {
        FunctionDescriptor fd = LinuxSyscallSignatures.all().get(name);
        if (fd == null) {
            throw new IllegalStateException("no signature for libc symbol: " + name);
        }
        return LINKER.downcallHandle(
                LIBC.find(name).orElseThrow(() -> new IllegalStateException("libc symbol not found: " + name)),
                fd, LinuxSyscallSignatures.CAPTURE);
    }

    static int errnoOf(MemorySegment capture) {
        return (int) ERRNO_VH.get(capture, 0L);
    }

    static int socket(MemorySegment cap, int domain, int type, int protocol) throws Throwable {
        return (int) SOCKET.invokeExact(cap, domain, type, protocol);
    }

    static int bind(MemorySegment cap, int fd, MemorySegment addr, int addrlen) throws Throwable {
        return (int) BIND.invokeExact(cap, fd, addr, addrlen);
    }

    static int listen(MemorySegment cap, int fd, int backlog) throws Throwable {
        return (int) LISTEN.invokeExact(cap, fd, backlog);
    }

    static int accept4(MemorySegment cap, int fd, int flags) throws Throwable {
        return (int) ACCEPT4.invokeExact(cap, fd, MemorySegment.NULL, MemorySegment.NULL, flags);
    }

    static long recv(MemorySegment cap, int fd, MemorySegment buf, long len, int flags) throws Throwable {
        return (long) RECV.invokeExact(cap, fd, buf, len, flags);
    }

    static long send(MemorySegment cap, int fd, MemorySegment buf, long len, int flags) throws Throwable {
        return (long) SEND.invokeExact(cap, fd, buf, len, flags);
    }

    static long recvmsg(MemorySegment cap, int fd, MemorySegment msghdr, int flags) throws Throwable {
        return (long) RECVMSG.invokeExact(cap, fd, msghdr, flags);
    }

    static int close(MemorySegment cap, int fd) throws Throwable {
        return (int) CLOSE.invokeExact(cap, fd);
    }

    static int fcntl(MemorySegment cap, int fd, int cmd, int arg) throws Throwable {
        return (int) FCNTL.invokeExact(cap, fd, cmd, arg);
    }

    static int setsockopt(MemorySegment cap, int fd, int level, int opt, MemorySegment val, int len)
            throws Throwable {
        return (int) SETSOCKOPT.invokeExact(cap, fd, level, opt, val, len);
    }

    static int epollCreate1(MemorySegment cap, int flags) throws Throwable {
        return (int) EPOLL_CREATE1.invokeExact(cap, flags);
    }

    static int epollCtl(MemorySegment cap, int epfd, int op, int fd, MemorySegment event) throws Throwable {
        return (int) EPOLL_CTL.invokeExact(cap, epfd, op, fd, event);
    }

    static int epollWait(MemorySegment cap, int epfd, MemorySegment events, int maxevents, int timeout)
            throws Throwable {
        return (int) EPOLL_WAIT.invokeExact(cap, epfd, events, maxevents, timeout);
    }

    static int unlink(MemorySegment cap, MemorySegment path) throws Throwable {
        return (int) UNLINK.invokeExact(cap, path);
    }

    static int chmod(MemorySegment cap, MemorySegment path, int mode) throws Throwable {
        return (int) CHMOD.invokeExact(cap, path, mode);
    }

    /** Allocate a per-thread errno capture segment from {@code arena}. */
    static MemorySegment newCapture(Arena arena) {
        return arena.allocate(CAPTURE);
    }

    private LinuxSyscalls() {
    }
}
