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

void VulkanContext::initWindow(ANativeWindow *platformWindow, uint32_t width, uint32_t height) {
    window.nativeWindow = platformWindow;
    window.width = width;
    window.height = height;
}

bool VulkanContext::initVulkan(AAssetManager *manager, bool enableDebug) {
    VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "vulkan_practice",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "vulkan_practice_engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_MAKE_VERSION(1, 1, 0),
    };

    // create device
    createVulkanDevice(enableDebug, &appInfo);

    createSwapChain();

    // create render pass
    createRenderPass();

    // Create 2 frame buffers.
    createFrameBuffers(render.renderPass_);

    // create vertex buffers
    createBuffers();

    // Create graphics pipeline
    createGraphicsPipeline(manager);

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
            // VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
            // VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            // VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
            // VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    if (enableDebug) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char *> deviceExtensions = {
            // VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
            // VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
            // VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            // VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
            // VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            // VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
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
    memset(&swapchain, 0, sizeof(swapchain));

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

void VulkanContext::createFrameBuffers(VkRenderPass &renderPass, VkImageView depthView) {
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
        VkImageView attachments[2] = {
                swapchain.displayViews_[i], depthView,
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
        fbCreateInfo.attachmentCount = (depthView == VK_NULL_HANDLE ? 1 : 2);

        CALL_VK(vkCreateFramebuffer, device.device_, &fbCreateInfo, nullptr,
                &swapchain.framebuffers_[i]);
    }

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
            -1.0f, -1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
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
    memset(&gfxPipeline, 0, sizeof(gfxPipeline));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
    };
    CALL_VK(vkCreatePipelineLayout, device.device_, &pipelineLayoutCreateInfo, nullptr,
            &gfxPipeline.layout_);

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
    VkVertexInputBindingDescription vertex_input_bindings{
            .binding = 0,
            .stride = 3 * sizeof(float),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    VkVertexInputAttributeDescription vertex_input_attributes[1] = {
            {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = 0,
            }
    };
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertex_input_bindings,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = vertex_input_attributes,
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
    VkGraphicsPipelineCreateInfo pipelineCreateInfo{
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
            .pDynamicState = nullptr,
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
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(render.cmdBuffer_[bufferIndex], 0, 1,
                               &buffers.vertexBuf_, &offset);

        // Draw Triangle
        vkCmdDraw(render.cmdBuffer_[bufferIndex], 3, 1, 0, 0);

        vkCmdEndRenderPass(render.cmdBuffer_[bufferIndex]);

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
