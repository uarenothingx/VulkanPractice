//
// Created by shenlinliang on 2023/5/9.
//

#ifndef VULKANPRACTICE_VULKANCONTEXT_H
#define VULKANPRACTICE_VULKANCONTEXT_H

#include <android/asset_manager_jni.h>
#include <android/hardware_buffer_jni.h>
#include <android/bitmap.h>
#include <vector>
#include "Utils.h"

class VulkanContext {

public:
    static std::unique_ptr<VulkanContext> create();

    void initWindow(ANativeWindow *platformWindow, uint32_t width, uint32_t height);

    bool initVulkan(JNIEnv *env,jobject bitmap, AHardwareBuffer *buffer, AAssetManager *manager, bool enableDebug);

    bool isVulkanReady();

    void unInitVulkan();

    void drawFrame();

private:
    struct WindowInfo {
        ANativeWindow *nativeWindow;
        uint32_t width;
        uint32_t height;
    };
    WindowInfo window = {};

    struct VulkanDeviceInfo {
        bool initialized_;

        VkInstance instance_;
        VkPhysicalDevice gpuDevice_;
        VkDevice device_;
        uint32_t queueFamilyIndex_;

        VkSurfaceKHR surface_;
        VkQueue queue_;
        VkPhysicalDeviceProperties deviceProperties_;
        VkPhysicalDeviceMemoryProperties deviceMemoryProperties_;
    };
    VulkanDeviceInfo device = {};

    struct VulkanSwapchainInfo {
        VkSwapchainKHR swapchain_;
        uint32_t swapchainLength_;

        VkExtent2D displaySize_;
        VkFormat displayFormat_;

        // array of frame buffers and views
        std::vector<VkImage> displayImages_;
        std::vector<VkImageView> displayViews_;
        std::vector<VkFramebuffer> framebuffers_;
    };
    VulkanSwapchainInfo swapchain = {};

    struct VulkanBufferInfo {
        VkBuffer vertexBuf_;
    };
    VulkanBufferInfo buffers = {};

    struct VulkanGfxPipelineInfo {
        VkDescriptorSetLayout dscLayout_;
        VkDescriptorPool descPool_;
        VkDescriptorSet descSet_;
        VkPipelineLayout layout_;
        VkPipelineCache cache_;
        VkPipeline pipeline_;
    };
    VulkanGfxPipelineInfo gfxPipeline = {};

    struct VulkanRenderInfo {
        VkRenderPass renderPass_;
        VkCommandPool cmdPool_;
        VkCommandBuffer *cmdBuffer_;
        uint32_t cmdBufferLen_;
        VkSemaphore semaphore_;
        VkFence fence_;
    };
    VulkanRenderInfo render = {};

    struct TextureObject {
        VkSampler sampler;
        VkImage image;
        VkDeviceMemory mem;
        VkImageView imageView;
        uint32_t texWidth;
        uint32_t texHeight;
    };
    TextureObject textureObject = {};

    struct HardwareObject {
        VkSampler sampler;
        VkImage image;
        VkDeviceMemory mem;
        VkImageView imageView;
    };
    HardwareObject hardwareObject = {};

    void createVulkanDevice(bool enableDebug, VkApplicationInfo *appInfo);

    void createRenderPass();

    void createSwapChain();

    void deleteSwapChain();

    void createFrameBuffers(VkRenderPass &renderPass);

    void createFenceAndSemaphore();

    bool createBuffers();

    void deleteBuffers();

    VkResult createGraphicsPipeline(AAssetManager *manager);

    void deleteGraphicsPipeline();

    void createCommandPool();

    void createTexture(JNIEnv* env, jobject bitmap);

    void createDescriptorSet();

    static bool mapMemoryTypeToIndex(VkPhysicalDeviceMemoryProperties memprop, uint32_t typeBits,
                                     VkFlags requirements_mask,
                                     uint32_t *typeIndex);

    static void setImageLayout(VkCommandBuffer cmdBuffer, VkImage image,
                               VkImageLayout oldImageLayout, VkImageLayout newImageLayout,
                               VkPipelineStageFlags srcStages,
                               VkPipelineStageFlags destStages);

    static VkResult buildShaderFromFile(AAssetManager *assetManager, const char *filePath,
                                        VkDevice device, VkShaderModule *shaderOut);

    static VkResult buildTextureFromBitmap(JNIEnv* env, jobject bitmap, VulkanDeviceInfo device,
                                           TextureObject* obj,VkImageUsageFlags usage);

    void createFromHardwareBuffer(AHardwareBuffer *buffer);
};

#endif //VULKANPRACTICE_VULKANCONTEXT_H
