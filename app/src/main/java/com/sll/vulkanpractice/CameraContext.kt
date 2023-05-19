package com.sll.vulkanpractice

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * @author Shenlinliang
 * @date 2023/5/16
 */
class CameraContext(private val applicationContext: Context) {
    /** Readers used as buffers for camera still shots */
    private lateinit var imageReader: ImageReader

    /** [HandlerThread] where all camera operations run */
    private lateinit var cameraThread: HandlerThread

    /** [Handler] corresponding to [cameraThread] */
    private lateinit var cameraHandler: Handler

    /** [HandlerThread] where all buffer reading operations run */
    private lateinit var imageReaderThread: HandlerThread

    /** [Handler] corresponding to [imageReaderThread] */
    private lateinit var imageReaderHandler: Handler

    /** The [CameraDevice] that will be opened in this fragment */
    private lateinit var camera: CameraDevice

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private lateinit var session: CameraCaptureSession

    private lateinit var targets: List<Surface>

    private var mOrientation: Int = 0

    fun init() {
        cameraThread = HandlerThread("CameraThread").apply { start() }
        cameraHandler = Handler(cameraThread.looper)
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread.looper)
    }

    suspend fun initializeCamera(
        configPreview: (prevSize: Size) -> Unit,
        onImageAvailable: (imageReader: ImageReader) -> Unit,
        previewSurface: Surface? = null
    ) {
        withContext(Dispatchers.Main) {
            // Open the selected camera
            val cameraManager =
                applicationContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            camera = openCamera(cameraManager, "1", cameraHandler)

            val characteristics = cameraManager.getCameraCharacteristics("1")

            mOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

            val previewSize = Size(1280, 720)

            Log.i("Vulkan", "previewSize:${previewSize.width}x${previewSize.height}")
            configPreview(previewSize)

            imageReader = ImageReader.newInstance(
                previewSize.width,
                previewSize.height,
                ImageFormat.PRIVATE,
                IMAGE_BUFFER_SIZE,
                HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE
            )

            imageReader.setOnImageAvailableListener(ImageReader.OnImageAvailableListener {
                onImageAvailable(it)
            }, imageReaderHandler)

            targets = if (previewSurface == null) {
                listOf(imageReader.surface)
            } else {
                listOf(imageReader.surface, previewSurface)
            }

            // Start a capture session using our open camera and list of Surfaces where frames will go
            session = createCaptureSession(camera, targets, cameraHandler)

            startPreview()
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(
        manager: CameraManager, cameraId: String, handler: Handler? = null
    ): CameraDevice = suspendCancellableCoroutine { cont ->
        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) = cont.resume(device)

            override fun onDisconnected(device: CameraDevice) {
                Log.w(TAG, "Camera $cameraId has been disconnected")
            }

            override fun onError(device: CameraDevice, error: Int) {
                val msg = when (error) {
                    ERROR_CAMERA_DEVICE -> "Fatal (device)"
                    ERROR_CAMERA_DISABLED -> "Device policy"
                    ERROR_CAMERA_IN_USE -> "Camera in use"
                    ERROR_CAMERA_SERVICE -> "Fatal (service)"
                    ERROR_MAX_CAMERAS_IN_USE -> "Maximum cameras in use"
                    else -> "Unknown"
                }
                val exc = RuntimeException("Camera $cameraId error: ($error) $msg")
                Log.e(TAG, exc.message, exc)
                if (cont.isActive) cont.resumeWithException(exc)
            }
        }, handler)
    }

    /**
     * Starts a [CameraCaptureSession] and returns the configured session (as the result of the
     * suspend coroutine
     */
    private suspend fun createCaptureSession(
        device: CameraDevice, targets: List<Surface>, handler: Handler? = null
    ): CameraCaptureSession = suspendCoroutine { cont ->

        // Create a capture session using the predefined targets; this also involves defining the
        // session state callback to be notified of when the session is ready
        device.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {

            override fun onConfigured(session: CameraCaptureSession) = cont.resume(session)

            override fun onConfigureFailed(session: CameraCaptureSession) {
                val exc = RuntimeException("Camera ${device.id} session configuration failed")
                Log.e(TAG, exc.message, exc)
                cont.resumeWithException(exc)
            }
        }, handler)
    }

    private fun startPreview() {
        val captureRequest = session.device.createCaptureRequest(
            CameraDevice.TEMPLATE_PREVIEW
        ).apply {
            for (item in targets) {
                addTarget(item)
            }
        }
        session.setRepeatingRequest(captureRequest.build(), null, cameraHandler)
    }

    public fun close() {
        camera.close()
    }

    companion object {
        private val TAG = "CameraContext"

        /** Maximum number of images that will be held in the reader's buffer */
        private const val IMAGE_BUFFER_SIZE: Int = 3
    }
}