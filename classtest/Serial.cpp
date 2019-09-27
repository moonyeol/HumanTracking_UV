#include <iostream>
#include <stdio.h>    /* Standard input/output definitions */
#include <string>
#include <string.h>   /* String function definitions */
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include "Serial.h"
 using namespace std;

//Serial_init::Serial_init(){
    //this->option();
//}

void Serial_init::init()
{
    fd=open("/dev/ttyACM0", O_RDWR | O_NOCTTY );  // 컨트롤 c 로 취소안되게 하기 | O_NOCTTY
    if (fd == -1)
    {
        open("/dev/ttyACM1", O_RDWR | O_NOCTTY );  // 컨트롤 c 로 취소안되게 하기 | O_NOCTTY
        if(fd == -1)
        {
            perror("init_serialport : Unable to open port ");
            return -1;
        }
    }
    return fd;
}

void Serial_init::option()

{
    struct termios toptions;
    toptions.c_cflag &= ~PARENB;//Enable parity generation on output and parity checking for input.

    toptions.c_cflag &= ~CSTOPB;//Set two stop bits, rather than one.

    toptions.c_cflag &= ~CSIZE;//Character size mask.  Values are CS5, CS6, CS7, or CS8.



    // no flow control

    toptions.c_cflag &= ~CRTSCTS;//(not in POSIX) Enable RTS/CTS (hardware) flow control. [requires _BSD_SOURCE or _SVID_SOURCE]

    toptions.c_cflag = B115200 | CS8 | CLOCAL | CREAD; // CLOCAL : Ignore modem control lines CREAD :Enable receiver.
    //toptions.c_cflag |= CREAD | CLOCAL;  // turn on READ & ignore ctrl lines

    toptions.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl
    toptions.c_iflag = IGNPAR | ICRNL;

    toptions.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Enable canonical mode (described below)./Echo input characters.
    // If ICANON is also set, the ERASE character erases the preced‐ing input character, and WERASE erases the preceding word.
    // When any of the characters INTR, QUIT, SUSP, or DSUSP are received, generate the corresponding signal.

    toptions.c_oflag &= ~OPOST; //Enable implementation-defined output processing.

    // see: http://unixwiz.net/techtips/termios-vmin-vtime.html
    toptions.c_cc[VMIN]  = 0;
    toptions.c_cc[VTIME] = 20;
}