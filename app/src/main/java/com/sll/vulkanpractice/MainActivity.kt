package com.sll.vulkanpractice

import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity
import com.sll.vulkanlib.NativeLib
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private lateinit var mSurfaceView: SurfaceView
    private val vulkan = NativeLib()
    private val isInit = AtomicBoolean(false)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

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
                    vulkan.init(assets)
                    draw()
                }
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.d(TAG, "[sll_debug] surfaceDestroyed: ")
                if (isInit.compareAndSet(true, false)) {
                    vulkan.deInit()
                }
            }
        })
    }

    private fun draw() {
        thread {
            while (isInit.get()) {
                vulkan.draw()
                Thread.sleep(1000)
            }
            Log.d(TAG, "[sll_debug] draw: done")
        }
    }
}