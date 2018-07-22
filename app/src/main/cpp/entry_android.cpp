#include "include/clpeak.h"

#define PRINT_CALLBACK   "print_callback_from_c"

extern "C"
JNIEXPORT void JNICALL
Java_com_sogou_xshuo_clpeak_MainActivity_setenv(JNIEnv *env, jobject instance, jstring key_,
                                                jstring value_) {
    const char *key = env->GetStringUTFChars(key_, 0);
    const char *value = env->GetStringUTFChars(value_, 0);

    setenv(key, value, 1);

    env->ReleaseStringUTFChars(key_, key);
    env->ReleaseStringUTFChars(value_, value);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_sogou_xshuo_clpeak_jni_1connect_launchClpeak(JNIEnv *_jniEnv, jobject _jObject, jint argc,
                                                      jobjectArray _argv) {

    char **argv;
    clPeak clObj;

    argv = (char **)malloc(sizeof(char*) * argc);

    // Convert jobjectArray to string array
    for(int i=0; i<argc; i++)
    {
      jstring strObj = (jstring) _jniEnv->GetObjectArrayElement(_argv, i);
      argv[i] = (char*) _jniEnv->GetStringUTFChars(strObj, 0);
    }
    clObj.parseArgs(argc, argv);

    if(argv)  free(argv);

    clObj.log->jEnv = _jniEnv;
    clObj.log->jObj = &(_jObject);
    clObj.log->printCallback = _jniEnv->GetMethodID(_jniEnv->GetObjectClass(_jObject),
                                                    PRINT_CALLBACK, "(Ljava/lang/String;)V");

    return clObj.runAll();
}