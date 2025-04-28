// #pragma once
#ifndef _DS_TIMER_H
#define _DS_TIMER_H

#include <string>   // std string

#ifndef UNIT
typedef unsigned int UNIT;
#endif

#ifndef _WIN32
    // For windows
    #include <Windows.h>
    typedef LARGE_INTEGER   TIME_VAL;
#else
    // For Unix/Linux
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/timeb.h>
    #include <string.h>     // c string
    typedef struct timeval  TIME_VAL;
#endif

#define TIMER_ON    true
#define TIMER_OFF   false


class DS_timer
{
private:

    bool turnOn;
    
    UNIT numTimer;
    UNIT numCounter;

    // For timers
    bool* timerStates;
    TIME_VAL* ticksPerSecond;
    TIME_VAL* start_ticks;
    TIME_VAL* end_ticks;
    TIME_VAL* totalTicks;

    char timerTitle[255];
    std::string* timerName;

    // For counters
    UNIT* counters;

    void memAllocCounters(void);
    void memAllocTimers(void);
    void releasecounters(void);
    void releaseTimers(void);

public:
    DS_timer(int _numTimer = 1, int _numCount = 1, bool _turnOn = true);
    ~DS_timer(void);

    // For configurations
    inline void timerOn(void) {
        turnOn = TIMER_ON;
    }
    inline void timerOff(void) {
        turnOn = TIMER_OFF;
    }

    UNIT getNumTimer(void);
    UNIT getNumCounter(void);
    UNIT setTimer(UNIT _numTimer);
    UNIT setCounter(UNIT _numCounter);

    // For timers

    void initTimer(UNIT id);
    void initTimers(void);
    void onTimer(UNIT id);
    void offTimer(UNIT id);
    double getTimer_ms(UNIT id);

    void setTimerTitle(char* _name) {
        memset(timerTitle, 0, sizeof(char)*255);
        memcpy(timerTitle, _name, strlen(_name));
    }

    void setTimerName(UNIT id, std::string &_name) {
        timerName[id] = _name;
    }
    void setTimerName(UNIT id, char* _name) {
        timerName[id] = _name;
    }

    // For counters

    void incCounter(UNIT id);
    void initCounters(void);
    void initCounter(UNIT id);
    void add2Counter(UNIT id, UNIT num);
    UNIT getCounter(UNIT id);

    // For reports
    void printTimer(float _denominator = 1);
    void printToFile(char* fileName, int _id = -1);
    void printTimerNameToFile(char* fileName);
};

#endif