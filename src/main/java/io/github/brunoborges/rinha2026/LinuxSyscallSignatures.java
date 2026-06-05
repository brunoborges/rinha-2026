package io.github.brunoborges.rinha2026;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.ValueLayout;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Side-effect-free catalogue of the libc downcall signatures used by {@link LinuxSyscalls}.
 *
 * <p>Kept separate from {@code LinuxSyscalls} on purpose: GraalVM Native Image must register every FFM
 * downcall at <em>build time</em> (via {@link ForeignDowncallFeature}), but the actual
 * {@link Linker#downcallHandle downcall handles} in {@code LinuxSyscalls} must be created at
 * <em>run time</em> in the native binary. Referencing this class from the build-time feature therefore
 * must not drag {@code LinuxSyscalls}'s static initializer (and its native handle creation) into the
 * image build. This holder has no native side effects, so both the feature and {@code LinuxSyscalls}
 * can share the exact same {@link FunctionDescriptor}s and capture {@link Linker.Option} with no drift.
 */
final class LinuxSyscallSignatures {

    private LinuxSyscallSignatures() {
    }

    /** Every handle captures {@code errno} as a prepended state segment; the feature must match this. */
    static final Linker.Option CAPTURE = Linker.Option.captureCallState("errno");

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG;
    private static final java.lang.foreign.AddressLayout PTR = ValueLayout.ADDRESS;

    /**
     * The ordered map of {@code libc symbol name -> downcall descriptor}. The order is irrelevant to
     * correctness; a {@link LinkedHashMap} just keeps it stable for logging.
     */
    static Map<String, FunctionDescriptor> all() {
        Map<String, FunctionDescriptor> m = new LinkedHashMap<>();
        m.put("socket", FunctionDescriptor.of(INT, INT, INT, INT));
        m.put("bind", FunctionDescriptor.of(INT, INT, PTR, INT));
        m.put("listen", FunctionDescriptor.of(INT, INT, INT));
        m.put("accept4", FunctionDescriptor.of(INT, INT, PTR, PTR, INT));
        m.put("recv", FunctionDescriptor.of(LONG, INT, PTR, LONG, INT));
        m.put("send", FunctionDescriptor.of(LONG, INT, PTR, LONG, INT));
        m.put("recvmsg", FunctionDescriptor.of(LONG, INT, PTR, INT));
        m.put("close", FunctionDescriptor.of(INT, INT));
        m.put("fcntl", FunctionDescriptor.of(INT, INT, INT, INT));
        m.put("setsockopt", FunctionDescriptor.of(INT, INT, INT, INT, PTR, INT));
        m.put("epoll_create1", FunctionDescriptor.of(INT, INT));
        m.put("epoll_ctl", FunctionDescriptor.of(INT, INT, INT, INT, PTR));
        m.put("epoll_wait", FunctionDescriptor.of(INT, INT, PTR, INT, INT));
        m.put("unlink", FunctionDescriptor.of(INT, PTR));
        m.put("chmod", FunctionDescriptor.of(INT, PTR, INT));
        return m;
    }
}
