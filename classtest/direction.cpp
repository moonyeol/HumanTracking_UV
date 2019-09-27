#include <iostream>
#include "direction.h"

using namespace std;

direction::direction()
{
   
}


char* direction::compare(long left, long right)
{
    size = right-left;
    xcenter = (right + left)/2;
    
    if (tempsize = 0)
    {
        tempsize = size;
    }
    else if (xcenter<100)
    {
        move = "l";
    }
    else if (xcenter > 520)
    {
        move = "r";
    }
    else if (tempsize < size-5)
    {
        tempsize = size;
        move = "b";
    }
    else if (tempsize > size+5)
    {
        tempsize = size;
        move = "g";
    }
    else
    {
        move = "s";
    }

    return move;
}