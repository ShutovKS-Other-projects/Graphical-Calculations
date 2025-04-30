// main.c  
#include <CL/cl.h>  
#include <stdio.h>  
#include <stdlib.h>  

#define ARRAY_SIZE 5 // Используем макрос для определения константного значения  

int main()  
{  
   const int arraySize = ARRAY_SIZE;  
   int a[ARRAY_SIZE] = { 1, 2, 3, 4, 5 };  
   int b[ARRAY_SIZE] = { 10, 20, 30, 40, 50 };  
   int c[ARRAY_SIZE] = { 0 };  

   cl_int err;  
   cl_platform_id platform;  
   cl_device_id device;  
   cl_context context;  
   cl_command_queue queue;  
   cl_program program;  
   cl_kernel kernel;  
   cl_mem bufA, bufB, bufC;  

   // 1. Платформа и устройство  
   err = clGetPlatformIDs(1, &platform, NULL);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clGetPlatformIDs -> %d\n", err); return 1; }  
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clGetDeviceIDs -> %d\n", err); return 1; }  

   // 2. Контекст и очередь  
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clCreateContext -> %d\n", err); return 1; }  

   cl_command_queue_properties properties[] = { 0 }; // Пустой массив свойств
   queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
   if (err != CL_SUCCESS) {
       fprintf(stderr, "clCreateCommandQueueWithProperties -> %d\n", err);
       return 1;
   }
   if (err != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue -> %d\n", err); return 1; }

   // 3. Читаем и билдим шейдер  
   const char* srcFilename = "add.cl";  
   FILE* f = fopen(srcFilename, "r");  
   if (!f) { perror("fopen"); return 1; }  
   fseek(f, 0, SEEK_END);  
   size_t sourceSize = ftell(f);  
   rewind(f);  
   char* sourceStr = malloc(sourceSize + 1);
   if (sourceStr == NULL) {
       fprintf(stderr, "Memory allocation failed\n");
       fclose(f);
       return 1;
   }
   fread(sourceStr, 1, sourceSize, f);
   sourceStr[sourceSize] = '\0';  
   fclose(f);  

   program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, &sourceSize, &err);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource -> %d\n", err); return 1; }  
   free(sourceStr);  

   err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);  
   if (err != CL_SUCCESS) {  
       // выведем лог сборки  
       size_t logSize;  
       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);  
       char* log = malloc(logSize);  
       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);  
       fprintf(stderr, "Build error:\n%s\n", log);  
       free(log);  
       return 1;  
   }  

   kernel = clCreateKernel(program, "addKernel", &err);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel -> %d\n", err); return 1; }  

   // 4. Буферы  
   bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
       ARRAY_SIZE * sizeof(int), a, &err);  
   bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
       ARRAY_SIZE * sizeof(int), b, &err);  
   bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  
       ARRAY_SIZE * sizeof(int), NULL, &err);  

   // 5. Аргументы ядра  
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);  
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);  
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);  

   // 6. Запуск  
   size_t globalWorkSize = ARRAY_SIZE;  
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,  
       &globalWorkSize, NULL, 0, NULL, NULL);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueNDRangeKernel -> %d\n", err); return 1; }  

   clFinish(queue);  

   // 7. Чтение результата  
   err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,  
       ARRAY_SIZE * sizeof(int), c, 0, NULL, NULL);  
   if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueReadBuffer -> %d\n", err); return 1; }  

   printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",  
       c[0], c[1], c[2], c[3], c[4]);  

   // 8. Cleanup  
   clReleaseMemObject(bufA);  
   clReleaseMemObject(bufB);  
   clReleaseMemObject(bufC);  
   clReleaseKernel(kernel);  
   clReleaseProgram(program);  
   clReleaseCommandQueue(queue);  
   clReleaseContext(context);  

   return 0;  
}
