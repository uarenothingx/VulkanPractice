#include <jni.h>
#include <string>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include "VulkanContext.h"

VulkanContext *castToProcessor(jlong handle) {
    return reinterpret_cast<VulkanContext *>(static_cast<uintptr_t>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_sll_vulkanlib_NativeLib_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


extern "C"
JNIEXPORT jlong JNICALL
Java_com_sll_vulkanlib_NativeLib_nativeInitVulkanEngine(JNIEnv *env, jobject thiz) {
    auto context = VulkanContext::create();
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(context.release()));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sll_vulkanlib_NativeLib_nativeInit(JNIEnv *env, jobject thiz, jlong engine,
                                            jobject bitmap,
                                            jobject buffer,
                                            jobject manager) {
    auto *assetManager = AAssetManager_fromJava(env, manager);
    AHardwareBuffer* nativeBuffer = AHardwareBuffer_fromHardwareBuffer(env, buffer);
    castToProcessor(engine)->initVulkan(env, bitmap, nativeBuffer, assetManager, false);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sll_vulkanlib_NativeLib_nativeOnSurfaceReady(JNIEnv *env, jobject thiz, jlong engine,
                                                      jobject surface, jint width, jint height) {
    auto window = ANativeWindow_fromSurface(env, surface);
    castToProcessor(engine)->initWindow(window, width, height);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sll_vulkanlib_NativeLib_nativeDraw(JNIEnv *env, jobject thiz, jlong engine) {
    auto context = castToProcessor(engine);
    if (context->isVulkanReady()) {
        context->drawFrame();
    }
}
extern "C"
JNIEXPORT void JNICALL
Java_com_sll_vulkanlib_NativeLib_nativeDeInit(JNIEnv *env, jobject thiz, jlong engine) {
    castToProcessor(engine)->unInitVulkan();
}