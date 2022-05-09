package com.chaquo.myapplication

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.IOException

class MainActivity : AppCompatActivity() {
    val mTestImages = arrayOf("test1_n.png", "test2.jpg", "test3.png")
    var mImageIndex = 0
    var mBitmap: Bitmap? = null
    var mImageView: ImageView? = null
    var mResultView: ImageView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)


        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                1
            )
        }
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }
        setContentView(R.layout.activity_main)
        try {
            mBitmap = BitmapFactory.decodeStream(assets.open(mTestImages.get(mImageIndex)))
        } catch (e: IOException) {
            Log.e("Object Detection", "Error reading assets", e)
            finish()
        }
        findViewById<ImageView>(R.id.imageView).setImageBitmap(mBitmap)
//        findViewById<ImageView>(R.id.imageView).setVisibility(View.INVISIBLE)
//
        val buttonTest = findViewById<Button>(R.id.testButton)
        buttonTest.text = "Test Image 1/3"
        buttonTest.setOnClickListener {
//            findViewById<ImageView>(R.id.imageView).setVisibility(View.INVISIBLE)
            mImageIndex = (mImageIndex + 1) % mTestImages.size
            buttonTest.text =
                String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.size)
            try {
                mBitmap = BitmapFactory.decodeStream(assets.open(mTestImages.get(mImageIndex)))
                findViewById<ImageView>(R.id.imageView).setImageBitmap(mBitmap)
            } catch (e: IOException) {
                Log.e("Object Detection", "Error reading assets", e)
                finish()
            }
        }
        val buttonSelect = findViewById<Button>(R.id.selectButton)
        buttonSelect.setOnClickListener {
//            findViewById<ImageView>(R.id.imageView).setVisibility(View.INVISIBLE)
            val options = arrayOf<CharSequence>("Choose from Photos", "Take Picture", "Cancel")
            val builder = AlertDialog.Builder(this@MainActivity)
            builder.setTitle("New Test Image")
            builder.setItems(options) { dialog, item ->
                if (options[item] == "Take Picture") {
                    val takePicture = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                    startActivityForResult(takePicture, 0)
//                    val photo = takePicture.extras!!["data"] as Bitmap?
//                    findViewById<ImageView>(R.id.imageView).setImageBitmap(photo)

                } else if (options[item] == "Choose from Photos") {
                    val pickPhoto = Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.INTERNAL_CONTENT_URI
                    )
                    startActivityForResult(pickPhoto, 1)
//                    val photo = pickPhoto.extras!!["data"] as Bitmap?
//                    findViewById<ImageView>(R.id.imageView).setImageBitmap(photo)
                } else if (options[item] == "Cancel") {
                    dialog.dismiss()
                }
            }
            builder.show()
        }
        val buttonLive = findViewById<Button>(R.id.liveButton)
        buttonLive.setOnClickListener {
            val intent = Intent(this@MainActivity, Live_button::class.java)
            startActivity(intent)
        }
        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        val module = py.getModule("plot")

        findViewById<Button>(R.id.button).setOnClickListener {
            try {
                val bytes = module.callAttr("plot").toJava(ByteArray::class.java)
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                findViewById<ImageView>(R.id.imageView).setImageBitmap(bitmap)

                currentFocus?.let {
                    (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
                        .hideSoftInputFromWindow(it.windowToken, 0)
                }
            } catch (e: PyException) {
                Toast.makeText(this, e.message, Toast.LENGTH_LONG).show()
            }
        }

    }
}