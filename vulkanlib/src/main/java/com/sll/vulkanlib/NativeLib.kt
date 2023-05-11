package com.sll.vulkanlib

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.view.Surface

class NativeLib {
    private var engine = nativeInitVulkanEngine()

    /**
     * A native method that is implemented by the 'vulkanlib' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    private external fun nativeInitVulkanEngine(): Long

    private external fun nativeInit(engine: Long, bitmap: Bitmap, manager: AssetManager)

    private external fun nativeOnSurfaceReady(
        engine: Long,
        surface: Surface,
        width: Int,
        height: Int
    )

    private external fun nativeDraw(engine: Long)

    private external fun nativeDeInit(engine: Long)

    fun init(bitmap: Bitmap, manager: AssetManager) {
        if (engine != -1L) {
            nativeInit(engine, bitmap, manager)
        }
    }

    fun onSurfaceReady(surface: Surface, width: Int, height: Int) {
        if (engine != -1L) {
            nativeOnSurfaceReady(engine, surface, width, height)
        }
    }

    fun draw() {
        if (engine != -1L) {
            nativeDraw(engine)
        }
    }

    fun deInit() {
        if (engine != -1L) {
            nativeDeInit(engine);
        }
    }

    companion object {
        // Used to load the 'vulkanlib' library on application startup.
        init {
            System.loadLibrary("vulkanlib")
        }
    }
}