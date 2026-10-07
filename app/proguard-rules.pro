# Default ProGuard rules for Android
# Add project specific ProGuard rules here.

# TFLite: Keep all TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# Keep YoloDetector and Detection inner class
-keep class com.example.usgandroidapp.YoloDetector { *; }
-keep class com.example.usgandroidapp.YoloDetector$Detection { *; }

