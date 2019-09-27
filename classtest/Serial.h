#ifndef SERIAL_H
#define SERIAL_H


class Serial_init
{
    public: 
        Serial_init();   
        int fd;
         void init();

    private: 
        struct termios toptions;
             
        void option();
       
};
#endif