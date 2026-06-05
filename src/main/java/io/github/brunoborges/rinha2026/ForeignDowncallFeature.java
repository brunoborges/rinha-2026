package io.github.brunoborges.rinha2026;

import java.lang.foreign.FunctionDescriptor;
import java.util.Map;

import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeForeignAccess;

/**
 * GraalVM Native Image feature that registers every libc downcall used by {@link LinuxSyscalls} so the
 * image build emits the corresponding {@code linkToNative} stubs. Without this, the analysis aborts with
 * {@code "unexpected input could not be handled: linkToNative"} and the build reports
 * {@code "0 downcalls registered for foreign access"} — the FFM downcall handles are created at run time
 * in the native binary and cannot be linked unless registered here at build time.
 *
 * <p>Enabled via {@code -H:Features=...ForeignDowncallFeature} (see the Dockerfile). This class is only
 * ever loaded by the image builder; it is not part of the runtime image.
 */
public final class ForeignDowncallFeature implements Feature {

    @Override
    public void duringSetup(DuringSetupAccess access) {
        for (Map.Entry<String, FunctionDescriptor> e : LinuxSyscallSignatures.all().entrySet()) {
            RuntimeForeignAccess.registerForDowncall(e.getValue(), LinuxSyscallSignatures.CAPTURE);
        }
    }
}
