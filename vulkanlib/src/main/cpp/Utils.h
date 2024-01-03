//
// Created by shenlinliang on 2023/5/8.
//

#ifndef VULKANPRACTICE_UTILS_H
#define VULKANPRACTICE_UTILS_H

#include <android/log.h>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_android.h>

#define LOG_TAG "VULKAN-PRACTICE"
#define LOG(severity, ...) ((void)__android_log_print(ANDROID_LOG_##severity, LOG_TAG,  __VA_ARGS__))
#define LOGE(...) LOG(ERROR, __VA_ARGS__)
#define LOGD(...) LOG(DEBUG, __VA_ARGS__)
#define LOGI(...) LOG(INFO, __VA_ARGS__)

// Invoke a Vulkan method, log an error and return false if the result is not VK_SUCCESS
#define CALL_VK(vkMethod, ...)                                                              \
    do {                                                                                    \
        auto _result = vkMethod(__VA_ARGS__);                                               \
        if (_result != VK_SUCCESS) {                                                        \
            LOGE("%s failed with %s at %s:%u", #vkMethod, vkResultToStr(_result), __FILE__, \
                 __LINE__);                                                                 \
        } else {                                                                            \
            LOGD("%s success", #vkMethod);                                                  \
        }                                                                                   \
    } while (0)

inline const char *vkResultToStr(VkResult result) {
    switch (result) {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_EVENT_SET:
            return "VK_EVENT_SET";
        case VK_EVENT_RESET:
            return "VK_EVENT_RESET";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:
            return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:
            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_SURFACE_LOST_KHR:
            return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR:
            return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:
            return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
            return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV:
            return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_FRAGMENTATION_EXT:
            return "VK_ERROR_FRAGMENTATION_EXT";
        case VK_ERROR_NOT_PERMITTED_EXT:
            return "VK_ERROR_NOT_PERMITTED_EXT";
        default:
            return "(Unknown VkResult)";
    }
}

#endif //VULKANPRACTICE_UTILS_H
