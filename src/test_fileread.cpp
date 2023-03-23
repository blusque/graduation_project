#include <iostream>
#include <stdio.h>
#include <time.h>
#include <Windows.h>

using namespace std;

char *getLine(char *buf, int &len)
{
    char *tmp = buf;
    if (*tmp == 'E')
        return NULL;
    while (*tmp && (*tmp != 0x0d && *tmp != 0x0a && *tmp != '\n'))
        ++tmp;
    // while(*tmp && (*tmp != 0x0d || *tmp != 0x0a )) ++tmp;
    len = tmp - buf; //

    // skip New-Line char
    if (*tmp == 0x0d)
    { // Windows style New-Line 0x0d 0x0a
        tmp += 2;
        // assert(*tmp == 0x0a);
    } // else Unix style New-Line 0x0a
    else
    {
        ++tmp;
    }

    return tmp;
}

int main(int argc, char **argv)
{

    clock_t start, finish;
    // strcpy(path, argv[1]); // 输入的文件名
    ios::sync_with_stdio(0);
    cin.tie(0);
    // 开始计时
    start = clock();
    HANDLE hFile = CreateFile("F:/Self_Study/graduation_project/senbai/data666.txt",
                              GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ,
                              NULL,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL,
                              NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        cout << " CreateFile fail" << endl;
        return -1;
    }

    // 创建一个文件映射内核对象
    HANDLE hFileMap = CreateFileMapping(hFile,
                                        NULL,
                                        PAGE_READWRITE,
                                        NULL,
                                        NULL,
                                        NULL);
    if (hFileMap == NULL)
    {
        cout << "CreateFileMapping fail" << endl;
        return -1;
    }

    // 将文件数据映射到进程的地址空间
    char *pMapData = (char *)MapViewOfFile(hFileMap,
                                           FILE_MAP_ALL_ACCESS,
                                           NULL,
                                           NULL,
                                           NULL);
    if (pMapData == NULL)
    {
        cout << " MapViewOfFile fail" << endl;
        return -1;
    }

    // 读取数据
    char *pBuf = pMapData;

    finish = clock();
    // cout << "Read 1g ID file time :" << float(finish - start) << "  ms " << endl;
    printf("Time cost on reading the file: %fms\n", (float)(finish - start));
    // 计时结束
    int index = 0;
    int len = 0;
    char *startBuf = pMapData;
    while (startBuf != nullptr)
    {
        startBuf = getLine(pBuf, len);
        std::cout << "index: " << index << std::endl;
        index++;
        pBuf = startBuf;
    }
    std::cout << "Index: " << index << std::endl;
    // for (int i = 0; i < 10000000; i++)
    // {
    //     std::cout << pBuf[i];
    // }

    // 撤销文件视图
    UnmapViewOfFile(pBuf);
    // 关闭映射文件句柄
    CloseHandle(hFile);
    CloseHandle(hFileMap);
    return 0;
}