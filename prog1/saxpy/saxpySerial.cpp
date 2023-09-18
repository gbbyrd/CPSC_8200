
void saxpySerial(int N,
                       float scale,
                       float X[],
                       float Y[],
                       float result[])
{

    for (int i=0; i<N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

void saxpySerial_v2(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    for (int j=0; j<1000; j++)
    {
        for (int i=0; i<N; i++) {
            result[i] = scale * X[i] + Y[i];
        }
    }
}

