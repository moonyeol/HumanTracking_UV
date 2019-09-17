/*

* Arduino-serial

* --------------

*

* A simple command-line example program showing how a computer can

* communicate with an Arduino board. Works on any POSIX system (Mac/Unix/PC)

*

*

* Compile with something like:

* gcc -o arduino-serial arduino-serial.c

*

* Created 5 December 2006

* Copyleft (c) 2006, Tod E. Kurt, tod@todbot.com

*

*

*

* Updated 8 December 2006:

*  Justin McBride discoevered B14400 & B28800 aren't in Linux's termios.h.

*  I've included his patch, but commented out for now.  One really needs a

*  real make system when doing cross-platform C and I wanted to avoid that

*  for this little program. Those baudrates aren't used much anyway. :)

*

* Updated 26 December 2007:

*  Added ability to specify a delay (so you can wait for Arduino Diecimila)

*  Added ability to send a binary byte number

*

*/


#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>    /* Standard input/output definitions */

#include <stdlib.h>

#include <stdint.h>   /* Standard types */

#include <string.h>   /* String function definitions */

#include <unistd.h>   /* UNIX standard function definitions */

#include <fcntl.h>    /* File control definitions */

#include <errno.h>    /* Error number definitions */

#include <termios.h>  /* POSIX terminal control definitions */

#include <sys/ioctl.h>

#include <getopt.h>
using namespace std;
using namespace cv;
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
char detectStatus = '0';
char temp = '0';
void detectAndDisplay(Mat frame);



void usage(void);

int serialport_init(const char* serialport, int baud);

int serialport_writebyte(int fd, uint8_t b);

int serialport_write(int fd, const char* str);

int serialport_read_until(int fd, char* buf, char until);



void usage(void) {

    printf("Usage: arduino-serial -p <serialport> [OPTIONS]\n"

           "\n"

           "Options:\n"

           "  -h, --help                   Print this help message\n"

           "  -p, --port=serialport        Serial port Arduino is on\n"

           "  -b, --baud=baudrate          Baudrate (bps) of Arduino\n"

           "  -s, --send=data              Send data to Arduino\n"

           "  -r, --receive                Receive data from Arduino & print it out\n"

           "  -n  --num=num                Send a number as a single byte\n"

           "  -d  --delay=millis           Delay for specified milliseconds\n"

           "\n"

           "Note: Order is important. Set '-b' before doing '-p'. \n"

           "      Used to make series of actions:  '-d 2000 -s hello -d 100 -r' \n"

           "      means 'wait 2secs, send 'hello', wait 100msec, get reply'\n"

           "\n");

}



int main(int argc, char *argv[])

{

    int fd = 0;

    char serialport[256];

    int baudrate = B115200;  // default

    char buf[20], dat[20], use[1];

    int rc,n;



    //baudrate = 9600;

    fd = serialport_init("/dev/ttyACM1", baudrate);

    if(fd==-1) {return -1;}
    else{
        std::cout<<"connect succesd"<<endl;
    }

    usleep(3000 * 1000 );

    if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
    //    if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    //    //-- 2. Read the video stream
    CvCapture.open(-1);
    if (!CvCapture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
    while (CvCapture.read(frame)) {
        if (frame.empty()) {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);

        int c = waitKey(10);
        if ((char) c == 27) { break; } // escape
    }

    while(1) {

        strcpy(dat, "00000000:\0");

        fgets(use, 8, stdin);

        if(use[0] == 'f') {

            dat[0] = 'f';

            dat[1] = 5;

        }

        else if(use[0] == 'k') {

            dat[0] = 'b';

            dat[1] = 5;

        }

        else if(use[0] == 'j') {

            dat[2] = 'f';

            dat[3] = 5;

        }

        else if(use[0] == 'l') {

            dat[2] = 'b';

            dat[3] = 5;

        }



        rc = serialport_write(fd, dat);

        if(rc==-1) return -1;

        //printf("Waiting until UART buffer clears: %d\n", tcdrain(fd));

        n = serialport_read_until(fd, buf, ':');

        printf("wrote %d bytes, read %d bytes: %s\n", rc, n, buf);

    }



    close(fd);



    exit(EXIT_SUCCESS);

} // end main



int serialport_writebyte( int fd, uint8_t b)

{

    int n = write(fd,&b,1);

    if( n!=1)

        return -1;

    return 0;

}



int serialport_write(int fd, const char* str)

{

    int len = strlen(str);

    int n = write(fd, str, len);

    if( n!=len )

        return -1;

    return n;

}



int serialport_read_until(int fd, char* buf, char until)

{

    char b[1];

    int i=0;

    do {

        int n = read(fd, b, 1);  // read a char at a time

        if( n==-1) return -1;    // couldn't read

        if( n==0 ) {

            usleep( 10 * 1000 ); // wait 10 msec try again

            continue;

        }

        buf[i] = b[0]; i++;

    } while( b[0] != until );



    buf[i] = 0;  // null terminate the string

    return i;

}



// takes the string name of the serial port (e.g. "/dev/tty.usbserial","COM1")

// and a baud rate (bps) and connects to that port at that speed and 8N1.

// opens the port in fully raw mode so you can send binary data.

// returns valid fd, or -1 on error

int serialport_init(const char* serialport, int baud)

{

    struct termios toptions;

    int fd;



    //fprintf(stderr,"init_serialport: opening port %s @ %d bps\n",

    //        serialport,baud);



    //fd = open(serialport, O_RDWR | O_NOCTTY | O_NDELAY);

    fd = open(serialport, O_RDWR | O_NOCTTY);

    if (fd == -1)  {

        perror("init_serialport: Unable to open port ");

        return -1;

    }



    if (tcgetattr(fd, &toptions) < 0) {

        perror("init_serialport: Couldn't get term attributes");

        return -1;

    }

    speed_t brate = baud; // let you override switch below if needed

    switch(baud) {

        case 4800:   brate=B4800;   break;

        case 9600:   brate=B9600;   break;

// if you want these speeds, uncomment these and set #defines if Linux

//#ifndef OSNAME_LINUX

//    case 14400:  brate=B14400;  break;

//#endif

        case 19200:  brate=B19200;  break;

//#ifndef OSNAME_LINUX

//    case 28800:  brate=B28800;  break;

//#endif

            //case 28800:  brate=B28800;  break;

        case 38400:  brate=B38400;  break;

        case 57600:  brate=B57600;  break;

        case 115200: brate=B115200; break;

    }

    cfsetispeed(&toptions, brate);

    cfsetospeed(&toptions, brate);



    // 8N1

    toptions.c_cflag &= ~PARENB;

    toptions.c_cflag &= ~CSTOPB;

    toptions.c_cflag &= ~CSIZE;

    toptions.c_cflag |= CS8;

    // no flow control

    toptions.c_cflag &= ~CRTSCTS;



    toptions.c_cflag |= CREAD | CLOCAL;  // turn on READ & ignore ctrl lines

    toptions.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl



    toptions.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // make raw

    toptions.c_oflag &= ~OPOST; // make raw



    // see: http://unixwiz.net/techtips/termios-vmin-vtime.html

    toptions.c_cc[VMIN]  = 0;

    toptions.c_cc[VTIME] = 20;



    if( tcsetattr(fd, TCSANOW, &toptions) < 0) {

        perror("init_serialport: Couldn't set term attributes");

        return -1;

    }



    return fd;

}
void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    if (faces.size() > 0){

        detectStatus = '1';
        if (temp == '0') {
            cout << "detect!!" << endl;
            serialport_write(1,buf);
            temp = '1';
        }
    }
    else {
        detectStatus = '0';
        if (temp == '1') {
            cout << "Unknown" << endl;
            serialport_write(0,)
            temp = '0';
        }
    }
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        //ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
        rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 4, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            //circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
            rectangle(frame, Point(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y), Point(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height), Scalar(255, 0, 0), 4, 8, 0);
        }

    }
    //-- Show what you got
    imshow(window_name, frame);
}
