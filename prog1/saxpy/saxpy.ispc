
export void saxpy_ispc(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    foreach (i = 0 ... N) {           
        result[i] = scale * X[i] + Y[i];
    }
}

task void saxpy_ispc_task(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    foreach (i = indexStart ... indexEnd) {
        result[i] = scale * X[i] + Y[i];
    }
}

task void saxpy_ispc_task_v2(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);
    for (int j=0; j<1000; j++)
    {
        foreach (i = indexStart ... indexEnd) {
            result[i] = scale * X[i] + Y[i];
        }
    }
}

export void saxpy_ispc_test_2(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    for (int j=0; j < 1000; j++) {
        for (int i = 0; i < N; i++) {           
            result[i] = scale * X[i] + Y[i];
        }
    }
}

export void saxpy_ispc_test_1(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    for (uniform int j=0; j < 1000; j++) {
        foreach (i = 0 ... N) {           
            result[i] = scale * X[i] + Y[i];
        }
    }
}

export void saxpy_ispc_withtasks(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{

    uniform int span = N / 8;  // 64 tasks

    launch[N/span] saxpy_ispc_task(N, span, scale, X, Y, result);
}

export void saxpy_ispc_withtasks_v2(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{

    uniform int span = N / 8;  // 64 tasks

    launch[8] saxpy_ispc_task_v2(N, span, scale, X, Y, result);
}


