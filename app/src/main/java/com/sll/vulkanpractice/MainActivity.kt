package com.sll.vulkanpractice

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity
import com.sll.vulkanlib.NativeLib
import java.util.concurrent.atomic.AtomicBoolean
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private lateinit var mSurfaceView: SurfaceView
    private val vulkan = NativeLib()
    private val isInit = AtomicBoolean(false)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//        val bitmap = BitmapFactory.decodeStream(assets.open("render-2992x4000.jpg"))
        val bitmap = BitmapFactory.decodeStream(assets.open("render-bitmap.jpg"))
//        val bitmap = BitmapFactory.decodeStream(assets.open("sample_tex.png"))
        assert(bitmap != null)

        mSurfaceView = findViewById(R.id.render_view)

        mSurfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                Log.d(TAG, "[sll_debug] surfaceCreated: ")
            }

            override fun surfaceChanged(
                holder: SurfaceHolder, format: Int, width: Int, height: Int
            ) {
                Log.d(TAG, "[sll_debug] surfaceChanged: ")
                if (isInit.compareAndSet(false, true)) {
                    vulkan.onSurfaceReady(holder.surface, width, height)
                    vulkan.init(bitmap, assets)
//                    vulkan.init(bitmap,  assets)
                }
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.d(TAG, "[sll_debug] surfaceDestroyed: ")
                if (isInit.compareAndSet(true, false)) {
                    vulkan.deInit()
                }
            }
        })



        val camera = CameraContext(applicationContext).apply { init() }
        lifecycleScope.launch {
            camera.initializeCamera({}, {
                val image = it.acquireLatestImage() ?: return@initializeCamera
                val buffer = image.hardwareBuffer ?: return@initializeCamera
                if (isInit.get()) {
                    vulkan.prepareHardwareBuffer(buffer)
                    vulkan.draw()
                }
                buffer.close()
                image.close()
            })
        }
    }

    private fun draw() {
        thread {
                vulkan.draw()
//            while (isInit.get()) {
//                vulkan.draw()
//                Thread.sleep(1000)
//            }
            Log.d(TAG, "[sll_debug] draw: done")
        }
    }
}