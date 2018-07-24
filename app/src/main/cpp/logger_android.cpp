#include "include/logger.h"
#include <iomanip>
#include <sstream>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

void logger::print(string str)
{
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(str.c_str()));
}

void logger::print(double val)
{
  stringstream ss;
  ss << setprecision(2) << fixed << val;

  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(float val)
{
  stringstream ss;
  ss << setprecision(2) << fixed << val;

  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(int val)
{
  stringstream ss;
  ss << val;

  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(unsigned int val)
{
  stringstream ss;
  ss << val;

  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}


logger::logger(bool _enableXml, string _xmlFileName): enableXml(_enableXml)
{
    if(enableXml)
    {
        xmlFile.open(_xmlFileName);
        xw = new xmlWriter(xmlFile);
        xmlFile.flush();
    }
}

logger::~logger()
{
    if(enableXml)
    {
        xw->closeAll();
        delete xw;
        xmlFile.close();
    }
}

// xshuo enabled
void logger::xmlOpenTag(string tag)
{
    if(enableXml)
    {
        xw->openElt(tag.c_str());
        xmlFile.flush();
    }
}

void logger::xmlAppendAttribs(string key, string value)
{
    if(enableXml)
    {
        xw->attr(key.c_str(), value.c_str());
        xmlFile.flush();
    }
}

void logger::xmlAppendAttribs(string key, uint value)
{
    if(enableXml)
    {
        stringstream ss;
        ss << value;

        xw->attr(key.c_str(), ss.str().c_str());
        xmlFile.flush();
    }
}

void logger::xmlSetContent(string value)
{
    if(enableXml)
    {
        xw->content(value.c_str());
        xmlFile.flush();
    }
}

void logger::xmlSetContent(float value)
{
    if(enableXml)
    {
        stringstream ss;
        ss << value;

        xw->content(ss.str().c_str());
        xmlFile.flush();
    }
}

void logger::xmlCloseTag()
{
    if(enableXml)
    {
        xw->closeElt();
        xmlFile.flush();
    }
}

void logger::xmlRecord(string tag, string value)
{
    if(enableXml)
    {
        stringstream ss;
        ss << value;

        xw->openElt(tag.c_str());
        xw->content(ss.str().c_str());
        xw->closeElt();
        xmlFile.flush();
    }
}

void logger::xmlRecord(string tag, float value)
{
    if(enableXml)
    {
        stringstream ss;
        ss << value;

        xw->openElt(tag.c_str());
        xw->content(ss.str().c_str());
        xw->closeElt();
        xmlFile.flush();
    }
}

void logger::log2File(const float *data, int M, int N) {
    ofstream f_output_stream;
    f_output_stream.open("/sdcard/log_Gemm1Row.txt");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            f_output_stream << *(data + i * N + j) << " ";
        }
        f_output_stream << std::endl;
    }
    //f_output_stream << "=======================" << std::endl;
    f_output_stream.flush();
    f_output_stream.close();
}
