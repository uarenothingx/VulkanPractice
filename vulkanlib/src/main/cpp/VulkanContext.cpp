//
// Created by shenlinliang on 2023/5/9.
//

#include <cassert>
#include "VulkanContext.h"

std::unique_ptr<VulkanContext> VulkanContext::create() {
    auto vk = std::make_unique<VulkanContext>();
    return std::move(vk);
}

bool VulkanContext::mapMemoryTypeToIndex(VkPhysicalDeviceMemoryProperties memprop,
                                         uint32_t typeBits, VkFlags requirements_mask,
                                         uint32_t *typeIndex) {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < memprop.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memprop.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *typeIndex = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    // No memory types matched, return failure
    return false;
}

void VulkanContext::setImageLayout(VkCommandBuffer cmdBuffer, VkImage image,
                                   VkImageLayout oldImageLayout, VkImageLayout newImageLayout,
                                   VkPipelineStageFlags srcStages,
                                   VkPipelineStageFlags destStages) {
    VkImageMemoryBarrier imageMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = 0,
            .oldLayout = oldImageLayout,
            .newLayout = newImageLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange =
                    {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    switch (oldImageLayout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        default:
            break;
    }

    switch (newImageLayout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask =
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            break;

        default:
            break;
    }

    vkCmdPipelineBarrier(cmdBuffer, srcStages, destStages, 0, 0, nullptr, 0, nullptr, 1,
                         &imageMemoryBarrier);
}

VkResult VulkanContext::buildShaderFromFile(AAssetManager *assetManager, const char *filePath,
                                            VkDevice device,
                                            VkShaderModule *shaderOut) {
    AAsset *shaderFile = AAssetManager_open(assetManager, filePath, AASSET_MODE_BUFFER);
    assert(shaderFile != nullptr);
    const size_t shaderSize = AAsset_getLength(shaderFile);
    std::vector<char> shader(shaderSize);
    int status = AAsset_read(shaderFile, shader.data(), shaderSize);
    AAsset_close(shaderFile);
    assert(status >= 0);

    VkShaderModuleCreateInfo shaderModuleCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = shaderSize,
            .pCode = (const uint32_t *) shader.data(),
    };

    VkResult result = vkCreateShaderModule(
            device, &shaderModuleCreateInfo, nullptr, shaderOut);
    return result;
}

VkResult VulkanContext::buildTextureFromBitmap(JNIEnv *env, jobject bitmap, VulkanDeviceInfo device,
                                               TextureObject *obj, VkImageUsageFlags usage
) {
    // Get bitmap info
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Image::createFromBitmap: Failed to AndroidBitmap_getInfo");
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    obj->texWidth = info.width;
    obj->texHeight = info.height;
    LOGD("width = %d, height = %d, stride = %d", info.width, info.height, info.stride);

    // Allocate the linear texture so texture could be copied over
    VkImageCreateInfo image_create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {
                    .width = obj->texWidth,
                    .height = obj->texHeight,
                    .depth = 1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &device.queueFamilyIndex_,
            .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
    };
    VkMemoryAllocateInfo mem_alloc = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = 0,
            .memoryTypeIndex = 0,
    };
    CALL_VK(vkCreateImage, device.device_, &image_create_info, nullptr, &obj->image);
    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(device.device_, obj->image, &mem_reqs);
    mem_alloc.allocationSize = mem_reqs.size;
    LOGI("allocationSize = %llu", mem_alloc.allocationSize);

    mapMemoryTypeToIndex(device.deviceMemoryProperties_, mem_reqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &mem_alloc.memoryTypeIndex);
    CALL_VK(vkAllocateMemory, device.device_, &mem_alloc, nullptr, &obj->mem);
    CALL_VK(vkBindImageMemory, device.device_, obj->image, obj->mem, 0);

    const uint32_t bufferSize = info.stride * info.height;

    // Create buffer
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    const VkBufferCreateInfo bufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    CALL_VK(vkCreateBuffer, device.device_, &bufferCreateInfo, nullptr, &staging);
    // Allocate memory for the buffer
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device.device_, staging, &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = 0,
    };
    mapMemoryTypeToIndex(device.deviceMemoryProperties_, memoryRequirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &allocateInfo.memoryTypeIndex);
    CALL_VK(vkAllocateMemory, device.device_, &allocateInfo, nullptr, &stagingMem);
    vkBindBufferMemory(device.device_, staging, stagingMem, 0);

    // Copy bitmap pixels to the buffer memory
    void *bitmapData = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &bitmapData);


    void *data;

    CALL_VK(vkMapMemory, device.device_, stagingMem, 0, mem_alloc.allocationSize, 0, &data);
    memcpy(data, bitmapData, bufferSize);
    vkUnmapMemory(device.device_, stagingMem);

    AndroidBitmap_unlockPixels(env, bitmap);

    VkCommandPoolCreateInfo cmdPoolCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.queueFamilyIndex_,
    };
    VkCommandPool cmdPool;
    CALL_VK(vkCreateCommandPool, device.device_, &cmdPoolCreateInfo, nullptr, &cmdPool);

    VkCommandBuffer gfxCmd;
    const VkCommandBufferAllocateInfo cmd = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = cmdPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
    };
    CALL_VK(vkAllocateCommandBuffers, device.device_, &cmd, &gfxCmd);

    VkCommandBufferBeginInfo cmd_buf_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr};
    CALL_VK(vkBeginCommandBuffer, gfxCmd, &cmd_buf_info);

    const VkBufferImageCopy bufferImageCopy = {
            .bufferOffset = 0,
            .bufferRowLength = info.stride / 4,
            .bufferImageHeight = info.height,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {info.width, info.height, 1},
    };
    vkCmdCopyBufferToImage(gfxCmd, staging, obj->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);

    CALL_VK(vkEndCommandBuffer, gfxCmd);
    VkFenceCreateInfo fenceInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };
    VkFence fence;
    CALL_VK(vkCreateFence, device.device_, &fenceInfo, nullptr, &fence);

    VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &gfxCmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
    };
    CALL_VK(vkQueueSubmit, device.queue_, 1, &submitInfo, fence);
    CALL_VK(vkWaitForFences, device.device_, 1, &fence, VK_TRUE, 100000000);
    vkDestroyFence(device.device_, fence, nullptr);

    vkFreeCommandBuffers(device.device_, cmdPool, 1, &gfxCmd);
    vkDestroyCommandPool(device.device_, cmdPool, nullptr);
    return VK_SUCCESS;
}

void VulkanContext::initWindow(ANativeWindow *platformWindow, uint32_t width, uint32_t height) {
    window.nativeWindow = platformWindow;
    window.width = width;
    window.height = height;
}

bool
VulkanContext::initVulkan(JNIEnv *env, jobject bitmap, AHardwareBuffer *buffer,
                          AAssetManager *manager, bool enableDebug) {
    VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "vulkan_practice",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "vulkan_practice_engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_MAKE_VERSION(1, 1, 0),
    };

    // Create Instance, Device And Pick QueueFamily<VK_QUEUE_GRAPHICS_BIT>
    createVulkanDevice(enableDebug, &appInfo);

    // Create Surface And Swapchain; Require NativeWindow<>
    createSwapChain();

    // Create render pass
    createRenderPass();

    // Create 2 frame buffers.
    createFrameBuffers(render.renderPass_);

    // Create bitmap texture
//    createTexture(env, bitmap);
    // todo render from hardware buffer
    LOGE("----- create hard ware buffer -----");
    createFromHardwareBuffer(buffer);
    LOGE("----- create hard ware buffer -----");

    // Create Vertex Fuffers
    createBuffers();

    // Create Fence And Semaphore; Require VkDevice
    createFenceAndSemaphore();

    // Create graphics pipeline
    createGraphicsPipeline(manager);

    createDescriptorSet();

    createCommandPool();

    device.initialized_ = true;

    return true;
}

bool VulkanContext::isVulkanReady() {
    return device.initialized_;
}

void VulkanContext::unInitVulkan() {
    vkFreeCommandBuffers(device.device_, render.cmdPool_, render.cmdBufferLen_,
                         render.cmdBuffer_);
    delete[] render.cmdBuffer_;

    vkDestroyCommandPool(device.device_, render.cmdPool_, nullptr);
    vkDestroyRenderPass(device.device_, render.renderPass_, nullptr);
    deleteSwapChain();
    deleteGraphicsPipeline();
    deleteBuffers();

    vkDestroyDevice(device.device_, nullptr);
    vkDestroyInstance(device.instance_, nullptr);

    device.initialized_ = false;
}


void VulkanContext::createVulkanDevice(bool enableDebug,
                                       VkApplicationInfo *appInfo) {
    std::vector<const char *> instanceLayers;
    if (enableDebug) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    }

    std::vector<const char *> instanceExtensions = {
             VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
             VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
             VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
             VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    if (enableDebug) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char *> deviceExtensions = {
             VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
             VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
             VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            // VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
             VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
             VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
            // VK_KHR_MAINTENANCE1_EXTENSION_NAME,
            // VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            // VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            // VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
    };

    // Create the Vulkan instance
    VkInstanceCreateInfo instanceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .pApplicationInfo = appInfo,
            .enabledLayerCount = static_cast<uint32_t>(instanceLayers.size()),
            .ppEnabledLayerNames = instanceLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
            .ppEnabledExtensionNames = instanceExtensions.data(),
    };

    CALL_VK(vkCreateInstance, &instanceCreateInfo, nullptr, &device.instance_);

    // Find one GPU to use:
    // On Android, every GPU device is equal -- supporting
    // graphics/compute/present
    uint32_t gpuCount = 0;
    CALL_VK(vkEnumeratePhysicalDevices, device.instance_, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> devices(gpuCount);
    CALL_VK(vkEnumeratePhysicalDevices, device.instance_, &gpuCount, devices.data());
    // Pick graphics GPU
    for (auto d: devices) {
        uint32_t numQueueFamilies = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &numQueueFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(numQueueFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &numQueueFamilies,
                                                 queueFamilies.data());
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                device.gpuDevice_ = d;
                device.queueFamilyIndex_ = i;
                break;
            }
        }
    }
    // Get GPU prop
    vkGetPhysicalDeviceProperties(device.gpuDevice_, &device.deviceProperties_);
    vkGetPhysicalDeviceMemoryProperties(device.gpuDevice_, &device.deviceMemoryProperties_);

    // Create a logical device (vulkan device)
    float priorities[] = {
            1.0f,
    };
    VkDeviceQueueCreateInfo queueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = device.queueFamilyIndex_,
            .queueCount = 1,
            .pQueuePriorities = priorities,
    };

    VkDeviceCreateInfo deviceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = nullptr,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = nullptr,
    };

    CALL_VK(vkCreateDevice, device.gpuDevice_, &deviceCreateInfo, nullptr,
            &device.device_);
    vkGetDeviceQueue(device.device_, device.queueFamilyIndex_, 0, &device.queue_);
}

void VulkanContext::createSwapChain() {
    // Create Surface
    VkAndroidSurfaceCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .window = window.nativeWindow};
    CALL_VK(vkCreateAndroidSurfaceKHR, device.instance_, &createInfo, nullptr,
            &device.surface_);

    // **********************************************************
    // Get the surface capabilities because:
    //   - It contains the minimal and max length of the chain, we will need it
    //   - It's necessary to query the supported surface format (R8G8B8A8 for
    //   instance ...)
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.gpuDevice_, device.surface_,
                                              &surfaceCapabilities);
    // Query the list of supported surface format and choose one we like
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.gpuDevice_, device.surface_,
                                         &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.gpuDevice_, device.surface_,
                                         &formatCount, formats.data());
    LOGI("Got %d formats", formatCount);
    uint32_t chosenFormat;
    for (chosenFormat = 0; chosenFormat < formatCount; chosenFormat++) {
        if (formats[chosenFormat].format == VK_FORMAT_R8G8B8A8_UNORM) break;
    }
    assert(chosenFormat < formatCount);
    LOGI("Chose %d format", chosenFormat);

    swapchain.displaySize_ = {
            .width = window.width,
            .height = window.height
    };
    swapchain.displayFormat_ = formats[chosenFormat].format;

    VkSurfaceCapabilitiesKHR surfaceCap;
    CALL_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR, device.gpuDevice_,
            device.surface_, &surfaceCap);
    assert(surfaceCap.supportedCompositeAlpha | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);

    // **********************************************************
    // Create a swap chain (here we choose the minimum available number of surface
    // in the chain)
    VkSwapchainCreateInfoKHR swapchainCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = nullptr,
            .surface = device.surface_,
            .minImageCount = surfaceCapabilities.minImageCount,
            .imageFormat = formats[chosenFormat].format,
            .imageColorSpace = formats[chosenFormat].colorSpace,
            .imageExtent = surfaceCapabilities.currentExtent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &device.queueFamilyIndex_,
            .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR,
            .clipped = VK_FALSE,
            .oldSwapchain = VK_NULL_HANDLE,
    };
    CALL_VK(vkCreateSwapchainKHR, device.device_, &swapchainCreateInfo, nullptr,
            &swapchain.swapchain_);

    // Get the length of the created swap chain
    CALL_VK(vkGetSwapchainImagesKHR, device.device_, swapchain.swapchain_,
            &swapchain.swapchainLength_, nullptr);

    LOGI("swapchainLength_ = %d", swapchain.swapchainLength_);
}

void VulkanContext::createRenderPass() {
    VkAttachmentDescription attachmentDescriptions{
            .format = swapchain.displayFormat_,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference colourReference = {
            .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpassDescription{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colourReference,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
    };

    VkRenderPassCreateInfo renderPassCreateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .attachmentCount = 1,
            .pAttachments = &attachmentDescriptions,
            .subpassCount = 1,
            .pSubpasses = &subpassDescription,
            .dependencyCount = 0,
            .pDependencies = nullptr,
    };
    CALL_VK(vkCreateRenderPass, device.device_, &renderPassCreateInfo, nullptr,
            &render.renderPass_);
}

void VulkanContext::deleteSwapChain() {
    for (int i = 0; i < swapchain.swapchainLength_; i++) {
        vkDestroyFramebuffer(device.device_, swapchain.framebuffers_[i], nullptr);
        vkDestroyImageView(device.device_, swapchain.displayViews_[i], nullptr);
    }
    vkDestroySwapchainKHR(device.device_, swapchain.swapchain_, nullptr);
}

void VulkanContext::createFrameBuffers(VkRenderPass &renderPass) {
    // query display attachment to swapchain
    uint32_t swapchainImagesCount = 0;
    CALL_VK(vkGetSwapchainImagesKHR, device.device_, swapchain.swapchain_,
            &swapchainImagesCount, nullptr);
    swapchain.displayImages_.resize(swapchainImagesCount);
    CALL_VK(vkGetSwapchainImagesKHR, device.device_, swapchain.swapchain_,
            &swapchainImagesCount,
            swapchain.displayImages_.data());

    // create image view for each swapchain image
    swapchain.displayViews_.resize(swapchainImagesCount);

    for (uint32_t i = 0; i < swapchainImagesCount; i++) {
        VkImageViewCreateInfo viewCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = swapchain.displayImages_[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchain.displayFormat_,
                .components =
                        {
                                .r = VK_COMPONENT_SWIZZLE_R,
                                .g = VK_COMPONENT_SWIZZLE_G,
                                .b = VK_COMPONENT_SWIZZLE_B,
                                .a = VK_COMPONENT_SWIZZLE_A,
                        },
                .subresourceRange =
                        {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };
        CALL_VK(vkCreateImageView, device.device_, &viewCreateInfo, nullptr,
                &swapchain.displayViews_[i]);
    }

    // create a framebuffer from each swapchain image
    swapchain.framebuffers_.resize(swapchain.swapchainLength_);
    for (uint32_t i = 0; i < swapchain.swapchainLength_; i++) {
        VkImageView attachments[1] = {
                swapchain.displayViews_[i],
        };
        VkFramebufferCreateInfo fbCreateInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = nullptr,
                .renderPass = renderPass,
                .attachmentCount = 1,  // 2 if using depth
                .pAttachments = attachments,
                .width = static_cast<uint32_t>(swapchain.displaySize_.width),
                .height = static_cast<uint32_t>(swapchain.displaySize_.height),
                .layers = 1,
        };
        fbCreateInfo.attachmentCount = 1;

        CALL_VK(vkCreateFramebuffer, device.device_, &fbCreateInfo, nullptr,
                &swapchain.framebuffers_[i]);
    }
}

void VulkanContext::createFenceAndSemaphore() {
    // We need to create a fence to be able, in the main loop, to wait for our
    // draw command(s) to finish before swapping the framebuffers
    VkFenceCreateInfo fenceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };
    CALL_VK(vkCreateFence, device.device_, &fenceCreateInfo, nullptr, &render.fence_);

    // We need to create a semaphore to be able to wait, in the main loop, for our
    // framebuffer to be available for us before drawing.
    VkSemaphoreCreateInfo semaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };
    CALL_VK(vkCreateSemaphore, device.device_, &semaphoreCreateInfo, nullptr,
            &render.semaphore_);
}


bool VulkanContext::createBuffers() {
    // Vertex positions
    const float vertexData[] = {
            -1.0f, -1.0f,
            0.0f, 0.0f, 0.0f,

            1.0f, -1.0f,
            0.0f, 1.0f, 0.0f,

            -1.0f, 1.0f,
            0.0f, 0.0f, 1.0f,

            1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, -1.0f,
            0.0f, 1.0f, 0.0f,

            -1.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
    };

    // Create a vertex buffer
    VkBufferCreateInfo createBufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = sizeof(vertexData),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &device.queueFamilyIndex_,
    };

    CALL_VK(vkCreateBuffer, device.device_, &createBufferInfo, nullptr,
            &buffers.vertexBuf_);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device.device_, buffers.vertexBuf_, &memReq);

    VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = memReq.size,
            .memoryTypeIndex = 0,  // Memory type assigned in the next step
    };

    // Assign the proper memory type for that buffer
    mapMemoryTypeToIndex(device.deviceMemoryProperties_, memReq.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &allocInfo.memoryTypeIndex);

    // Allocate memory for the buffer
    VkDeviceMemory deviceMemory;
    CALL_VK(vkAllocateMemory, device.device_, &allocInfo, nullptr, &deviceMemory);

    void *data;
    CALL_VK(vkMapMemory, device.device_, deviceMemory, 0, allocInfo.allocationSize, 0, &data);
    memcpy(data, vertexData, sizeof(vertexData));
    vkUnmapMemory(device.device_, deviceMemory);

    CALL_VK(vkBindBufferMemory, device.device_, buffers.vertexBuf_, deviceMemory, 0);
    return true;
}

void VulkanContext::deleteBuffers() {
    vkDestroyBuffer(device.device_, buffers.vertexBuf_, nullptr);
}

VkResult VulkanContext::createGraphicsPipeline(AAssetManager *manager) {
    const VkDescriptorSetLayoutBinding descriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
    };
    const VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = 1,
            .pBindings = &descriptorSetLayoutBinding,
    };
    CALL_VK(vkCreateDescriptorSetLayout, device.device_,
            &descriptorSetLayoutCreateInfo, nullptr,
            &gfxPipeline.dscLayout_);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .setLayoutCount = 1,
            .pSetLayouts = &gfxPipeline.dscLayout_,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
    };
    CALL_VK(vkCreatePipelineLayout, device.device_, &pipelineLayoutCreateInfo, nullptr,
            &gfxPipeline.layout_);

    // No dynamic state in that tutorial
    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .dynamicStateCount = 0,
            .pDynamicStates = nullptr
    };

    VkShaderModule vertexShader, fragmentShader;
    buildShaderFromFile(manager, "shaders/tri.vert.spv", device.device_, &vertexShader);
    buildShaderFromFile(manager, "shaders/tri.frag.spv", device.device_, &fragmentShader);

    // Specify vertex and fragment shader stages
    VkPipelineShaderStageCreateInfo shaderStages[2] = {
            {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertexShader,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
            },
            {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragmentShader,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
            }
    };

    VkViewport viewports = {
            .x = 0,
            .y = 0,
            .width = (float) swapchain.displaySize_.width,
            .height = (float) swapchain.displaySize_.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
            .offset {.x = 0, .y = 0,},
            .extent = swapchain.displaySize_,
    };
    // Specify viewport info
    VkPipelineViewportStateCreateInfo viewportInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .viewportCount = 1,
            .pViewports = &viewports,
            .scissorCount = 1,
            .pScissors = &scissor,
    };

    // Specify multisample info
    VkSampleMask sampleMask = ~0u;
    VkPipelineMultisampleStateCreateInfo multisampleInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 0,
            .pSampleMask = &sampleMask,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
    };

    // Specify color blend state
    VkPipelineColorBlendAttachmentState attachmentStates{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo colorBlendInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachmentStates,
    };

    // Specify rasterizer info
    VkPipelineRasterizationStateCreateInfo rasterInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1,
    };

    // Specify input assembler state
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
    };

    // Specify vertex input state
    VkVertexInputBindingDescription vertex_input_bindings = {
            .binding = 0,
            .stride = 5 * sizeof(float),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = 0,
            },
            {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = sizeof(float) * 3,
            }
    };
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertex_input_bindings,
            .vertexAttributeDescriptionCount = 2,
            .pVertexAttributeDescriptions = vertex_input_attributes.data(),
    };

    // Create the pipeline cache
    VkPipelineCacheCreateInfo pipelineCacheInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,  // reserved, must be 0
            .initialDataSize = 0,
            .pInitialData = nullptr,
    };

    CALL_VK(vkCreatePipelineCache, device.device_, &pipelineCacheInfo, nullptr,
            &gfxPipeline.cache_);

    // Create the pipeline
    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pTessellationState = nullptr,
            .pViewportState = &viewportInfo,
            .pRasterizationState = &rasterInfo,
            .pMultisampleState = &multisampleInfo,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = gfxPipeline.layout_,
            .renderPass = render.renderPass_,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = 0,
    };

    VkResult pipelineResult = vkCreateGraphicsPipelines(
            device.device_, gfxPipeline.cache_, 1, &pipelineCreateInfo, nullptr,
            &gfxPipeline.pipeline_);

    // We don't need the shaders anymore, we can release their memory
    vkDestroyShaderModule(device.device_, vertexShader, nullptr);
    vkDestroyShaderModule(device.device_, fragmentShader, nullptr);

    return pipelineResult;
}

void VulkanContext::deleteGraphicsPipeline() {
    if (gfxPipeline.pipeline_ == VK_NULL_HANDLE) return;
    vkDestroyPipeline(device.device_, gfxPipeline.pipeline_, nullptr);
    vkDestroyPipelineCache(device.device_, gfxPipeline.cache_, nullptr);
    vkDestroyPipelineLayout(device.device_, gfxPipeline.layout_, nullptr);
}

void VulkanContext::createCommandPool() {
    // Create a pool of command buffers to allocate command buffer from
    VkCommandPoolCreateInfo cmdPoolCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.queueFamilyIndex_,
    };
    CALL_VK(vkCreateCommandPool, device.device_, &cmdPoolCreateInfo, nullptr,
            &render.cmdPool_);

    // Record a command buffer that just clear the screen
    // 1 command buffer draw in 1 framebuffer
    // In our case we need 2 command as we have 2 framebuffer
    render.cmdBufferLen_ = swapchain.swapchainLength_;
    render.cmdBuffer_ = new VkCommandBuffer[swapchain.swapchainLength_];
    VkCommandBufferAllocateInfo cmdBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = render.cmdPool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = render.cmdBufferLen_,
    };
    CALL_VK(vkAllocateCommandBuffers, device.device_, &cmdBufferCreateInfo,
            render.cmdBuffer_);

    const std::vector<VkClearValue> clearVals = {
            {
                    .color = {
                            .float32 {
                                    1.0f, 0.0f, 0.0f, 1.0f
                            }
                    }
            },
            {
                    .color = {
                            .float32 {
                                    0.0f, 1.0f, 0.0f, 1.0f
                            }
                    }
            },
            {
                    .color = {
                            .float32 {
                                    0.0f, 0.0f, 1.0f, 1.0f
                            }
                    }
            },
            {
                    .color = {
                            .float32 {
                                    1.0f, 1.0f, 0.0f, 1.0f
                            }
                    }
            }
    };

    for (int bufferIndex = 0; bufferIndex < swapchain.swapchainLength_;
         bufferIndex++) {
        // We start by creating and declare the "beginning" our command buffer
        VkCommandBufferBeginInfo cmdBufferBeginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pInheritanceInfo = nullptr,
        };
        CALL_VK(vkBeginCommandBuffer, render.cmdBuffer_[bufferIndex],
                &cmdBufferBeginInfo);

        // transition the display image to color attachment layout
        setImageLayout(render.cmdBuffer_[bufferIndex],
                       swapchain.displayImages_[bufferIndex],
                       VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        // Now we start a renderpass. Any draw command has to be recorded in a
        // renderpass
        VkRenderPassBeginInfo renderPassBeginInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = render.renderPass_,
                .framebuffer = swapchain.framebuffers_[bufferIndex],
                .renderArea = {.offset {.x = 0, .y = 0,},
                        .extent = swapchain.displaySize_},
                .clearValueCount = 1,
                .pClearValues = &clearVals[bufferIndex]};
        vkCmdBeginRenderPass(render.cmdBuffer_[bufferIndex], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        // Bind what is necessary to the command buffer
        vkCmdBindPipeline(render.cmdBuffer_[bufferIndex],
                          VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipeline.pipeline_);
        vkCmdBindDescriptorSets(
                render.cmdBuffer_[bufferIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                gfxPipeline.layout_, 0, 1, &gfxPipeline.descSet_, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(render.cmdBuffer_[bufferIndex], 0, 1,
                               &buffers.vertexBuf_, &offset);

        // Draw Triangle
        vkCmdDraw(render.cmdBuffer_[bufferIndex], 6, 1, 0, 0);

        vkCmdEndRenderPass(render.cmdBuffer_[bufferIndex]);
        setImageLayout(render.cmdBuffer_[bufferIndex],
                       swapchain.displayImages_[bufferIndex],
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        CALL_VK(vkEndCommandBuffer, render.cmdBuffer_[bufferIndex]);
    }
}

void VulkanContext::drawFrame() {
    uint32_t nextIndex;
    // Get the framebuffer index we should draw in
    CALL_VK(vkAcquireNextImageKHR, device.device_, swapchain.swapchain_,
            UINT64_MAX, render.semaphore_, VK_NULL_HANDLE,
            &nextIndex);

    CALL_VK(vkResetFences, device.device_, 1, &render.fence_);

    VkPipelineStageFlags waitStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render.semaphore_,
            .pWaitDstStageMask = &waitStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &render.cmdBuffer_[nextIndex],
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr
    };

    CALL_VK(vkQueueSubmit, device.queue_, 1, &submit_info, render.fence_);
    CALL_VK(vkWaitForFences, device.device_, 1, &render.fence_, VK_TRUE, 100000000);

    VkResult result;
    VkPresentInfoKHR presentInfo = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .swapchainCount = 1,
            .pSwapchains = &swapchain.swapchain_,
            .pImageIndices = &nextIndex,
            .pResults = &result,
    };
    vkQueuePresentKHR(device.queue_, &presentInfo);
}


void VulkanContext::createFromHardwareBuffer(AHardwareBuffer *buffer) {
    struct hwinfo {
        AHardwareBuffer *buffer = nullptr;
        uint64_t allocationSize = 0;
        uint32_t memoryTypeIndex = 0;
    } info;

    AHardwareBuffer_Desc ahwbDesc = {};
    AHardwareBuffer_describe(buffer, &ahwbDesc);

    info.buffer = buffer;

    VkAndroidHardwareBufferFormatPropertiesANDROID formatInfo = {
            .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
    };
    VkAndroidHardwareBufferPropertiesANDROID properties = {
            .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
            .pNext = &formatInfo,
    };
    CALL_VK(vkGetAndroidHardwareBufferPropertiesANDROID, device.device_, info.buffer, &properties);

    // init image
    {
        VkExternalMemoryImageCreateInfo externalCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID
        };
        VkExternalFormatANDROID extFormat = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
                .pNext = &externalCreateInfo,
        };

        VkImageCreateInfo image_create_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &extFormat,
                .imageType = VK_IMAGE_TYPE_2D,
                .extent = {
                        .width = ahwbDesc.width,
                        .height = ahwbDesc.height,
                        .depth = 1,
                },
                .mipLevels = 1u,
                .arrayLayers = ahwbDesc.layers,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices   = nullptr,
                .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        image_create_info.format = formatInfo.format;
        if (image_create_info.format == VK_FORMAT_UNDEFINED) {
            extFormat.externalFormat = formatInfo.externalFormat;
        }

        CALL_VK(vkCreateImage, device.device_, &image_create_info, nullptr, &hardwareObject.image);
    }

    // allocate mem
    {
        auto map = mapMemoryTypeToIndex(device.deviceMemoryProperties_,
                                        properties.memoryTypeBits, 0, &info.memoryTypeIndex);

        LOGE("mapMemoryTypeToIndex, %d", map);
        info.allocationSize = properties.allocationSize;

        VkImportAndroidHardwareBufferInfoANDROID androidHardwareBufferInfo = {
                .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                .buffer = info.buffer,
        };
        VkMemoryDedicatedAllocateInfo memoryAllocateInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .pNext  = &androidHardwareBufferInfo,
                .image  = hardwareObject.image,
        };
        VkMemoryAllocateInfo allocateInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext           = &memoryAllocateInfo,
                .allocationSize  = info.allocationSize,
                .memoryTypeIndex = info.memoryTypeIndex,
        };

        CALL_VK(vkAllocateMemory, device.device_, &allocateInfo, nullptr, &hardwareObject.mem);
    }

    // bind mem image
    {
        VkBindImageMemoryInfo bind_info = {
                .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                .image = hardwareObject.image,
                .memory = hardwareObject.mem,
                .memoryOffset = 0,
        };
        CALL_VK(vkBindImageMemory2, device.device_, 1, &bind_info);
    }

    // assert mem requires
    {
        VkImageMemoryRequirementsInfo2 mem_reqs_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                .image = hardwareObject.image
        };
        VkMemoryDedicatedRequirements ded_mem_reqs = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
        };
        VkMemoryRequirements2 mem_reqs2 = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                .pNext = &ded_mem_reqs,
        };
        vkGetImageMemoryRequirements2(device.device_, &mem_reqs_info, &mem_reqs2);
        if (!ded_mem_reqs.prefersDedicatedAllocation || !ded_mem_reqs.requiresDedicatedAllocation) {
            LOGE("mem error");
        }
    }

    // ycb sampler and image view
    {
        VkExternalFormatANDROID externalFormatAndroid = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
        };
        VkSamplerYcbcrConversionCreateInfo samplerYcbConversionDesc = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
                .pNext = &externalFormatAndroid,
                .format = formatInfo.format,
                .ycbcrRange = formatInfo.suggestedYcbcrRange,
                .components = formatInfo.samplerYcbcrConversionComponents,
                .xChromaOffset = formatInfo.suggestedXChromaOffset,
                .yChromaOffset = formatInfo.suggestedYChromaOffset,
                .chromaFilter = VK_FILTER_NEAREST,
                .forceExplicitReconstruction = false,
        };
        if (samplerYcbConversionDesc.format == VK_FORMAT_UNDEFINED) {
            externalFormatAndroid.externalFormat = formatInfo.externalFormat;
            samplerYcbConversionDesc.ycbcrModel = formatInfo.suggestedYcbcrModel;
        } else {
            samplerYcbConversionDesc.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
        }

        VkSamplerYcbcrConversion ycbConversion;
        CALL_VK(vkCreateSamplerYcbcrConversion, device.device_, &samplerYcbConversionDesc, nullptr, &ycbConversion);

        VkSamplerYcbcrConversionInfo ycbConversionDesc = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
                .conversion = ycbConversion,
        };
        VkSamplerCreateInfo samplerDesc = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext                   = &ycbConversionDesc,
                .magFilter               = VK_FILTER_NEAREST,
                .minFilter               = VK_FILTER_NEAREST,
                .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .mipLodBias              = 0.0f,
                .anisotropyEnable        = false,
                .maxAnisotropy           = 1.0f,
                .compareEnable           = false,
                .compareOp               = VK_COMPARE_OP_NEVER,
                .minLod                  = 0.0f,
                .maxLod                  = 0.0f,
                .borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
                .unnormalizedCoordinates = false,
        };

        CALL_VK(vkCreateSampler, device.device_, &samplerDesc, nullptr, &hardwareObject.sampler);

        VkImageViewCreateInfo imageViewDesc = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext                           = &ycbConversionDesc,
                .image                           = hardwareObject.image,
                .viewType                        = VK_IMAGE_VIEW_TYPE_2D,
                .format                          = formatInfo.format,
                .components                      = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .subresourceRange.baseMipLevel   = 0,
                .subresourceRange.levelCount     = 1,
                .subresourceRange.baseArrayLayer = 0,
                .subresourceRange.layerCount     = 1,
        };

        CALL_VK(vkCreateImageView, device.device_, &imageViewDesc, nullptr, &hardwareObject.imageView);
    }
}

void VulkanContext::createTexture(JNIEnv *env, jobject bitmap) {
    buildTextureFromBitmap(env, bitmap, device, &textureObject, VK_IMAGE_USAGE_SAMPLED_BIT);

    const VkSamplerCreateInfo sampler = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .maxAnisotropy = 1,
            .compareOp = VK_COMPARE_OP_NEVER,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
            .unnormalizedCoordinates = VK_FALSE,
    };
    VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = textureObject.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .components =
                    {
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };

    CALL_VK(vkCreateSampler, device.device_, &sampler, nullptr, &textureObject.sampler);
    CALL_VK(vkCreateImageView, device.device_, &view, nullptr, &textureObject.imageView);
}

void VulkanContext::createDescriptorSet() {
    const VkDescriptorPoolSize type_count = {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
    };
    const VkDescriptorPoolCreateInfo descriptor_pool = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &type_count,
    };
    CALL_VK(vkCreateDescriptorPool, device.device_, &descriptor_pool, nullptr,
            &gfxPipeline.descPool_);

    VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = gfxPipeline.descPool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &gfxPipeline.dscLayout_};
    CALL_VK(vkAllocateDescriptorSets, device.device_, &alloc_info,
            &gfxPipeline.descSet_);

    VkDescriptorImageInfo texDsts[1] = {
            {
                    .sampler = hardwareObject.sampler,
                    .imageView = hardwareObject.imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            }
    };

    VkWriteDescriptorSet writeDst = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = gfxPipeline.descSet_,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = texDsts,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr};
    vkUpdateDescriptorSets(device.device_, 1, &writeDst, 0, nullptr);
}
