int ComputeMySum(int tid)
{
    int my_sum[tid] = 0;

    for(int i = my_first; i < my_end; i++)
    {
        int my_x = ComptutNextValue();
        my_sum[tid] += my_x;

    }
}

void main(void)
{
    ComputeMySum(tid);  // tid: 스레드의 ID

    if(master_thread)
    {
        // 대표 스레드
        sum = my_x;
        for(int i = l; i < p; i++)
        {
            sum += receieve(i);
        }
    }
    else
    {
        // 대표 스레드 외외
        sendMySum();
    }
}