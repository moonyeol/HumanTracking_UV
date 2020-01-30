/* EON Embedded Contest! */

#include <iostream>
#include <rplidar.h>
#include <algorithm>    // 수학적 거리 계산 전용
#include <string.h>
#include <termios.h>

#define CYCLE 360
#define DIRECTION 4

#define GO      'g'
#define BACK    'b'
#define LEFT    'l'
#define RIGHT   'r'
#define STOP    's'

#define F       2
#define B       0
#define L       1
#define R       3

#define DIST_STOP 300   // 장애물 기준 거리를 500mm, 즉 0.5미터로 잡는다.
#define DIST_REF 500    // 방향을 틀었을 때, 최소한 0.7미터의 여유가 있을 때로 선택한다.

#include <thread>                           // THREADING
#include <unistd.h>                         // C & C++ LIBRARY FOR POSIX.
#include <fcntl.h>                          // POSIX FILE DESCRIPTOR OPERATIONS (read, write).
#include <sys/stat.h>                       // SET PERMISSION FOR CREATING FIFO.
#define FIFO_FROM_OPENCV "from_opencv"
int fd_from_opencv;

#define BUFF_SIZE   4
char buff[BUFF_SIZE];
char buff_a[BUFF_SIZE];
char move = 's';
bool isEnd = false;
#include <csignal>                          // C & C++ LIBRARY FOR UNIX SIGNAL.
bool rplidarSTOP = false;
void ctrlc(int){
    rplidarSTOP = true;
}

using namespace rp::standalone::rplidar;

void rplidarMEASURE(int [4], rplidar_response_measurement_node_hq_t*, size_t);
char rplidarMOVE(int*, char, int*, int*);
void rplidarRESULT(int [4], char);

/*
void fifo_from_read() {
    if ( -1 == (fd_from_opencv = open(FIFO_FROM_OPENCV, O_RDWR)) ){
        if ( mkfifo(FIFO_FROM_OPENCV, 0666) == -1 ){
            perror( "[ERROR] CANNOT CREATE FIFO");
            exit(1);
        }
        if ( -1 == ( fd_from_opencv = open(FIFO_FROM_OPENCV, O_RDWR)) ){
            perror("[ERROR] CANNOT OPEN FIFO");
            exit(1);
        }
    }
    while(1){
        if (rplidarSTOP) break;
        read(fd_from_opencv, buff, BUFF_SIZE);
        //std::cout << "READ " << buff[0] << ": ";
    }
    std::cout << "[FIFO] ENDING NOW." << std::endl;
}
*/

void fifo_from_read() {
    fd_from_opencv = open(FIFO_FROM_OPENCV, O_RDONLY | O_CREAT, 0666);
	while (1) {
		lseek(fd_from_opencv, 0, SEEK_CUR);
		if (-1 == read(fd_from_opencv, buff, BUFF_SIZE)) {
			perror("[ERROR] CANNOT READ");
			exit(1);
		}
		if (buff[0] == 'q') {
			isEnd = true;
			break;
		}
	}
	std::cout << "[FIFO] ENDING NOW." << std::endl;
	close(fd_from_opencv);
}




int main() {
int     handle;
	struct  termios  oldtio, newtio;

	// 화일을 연다.
	handle = open("/dev/ttyTHS0", O_RDWR | O_NOCTTY| O_TRUNC, 0666);
	if (handle < 0)
	{
		//화일 열기 실패
		printf("Serial Open Fail [/dev/ttyTHS0]\r\n ");
		handle = open("/dev/ttyTHS1", O_WRONLY | O_NOCTTY | O_TRUNC, 0666);
		if (handle < 0)
		{
		printf("Serial Open Fail [/dev/ttyTHS1]\r\n ");
		handle = open("/dev/ttyTHS4", O_WRONLY | O_NOCTTY | O_TRUNC, 0666);
			exit(1);
		}
	}
	tcgetattr(handle, &oldtio);  // 현재 설정을 oldtio에 저장
	memset(&newtio, 0, sizeof(newtio));
	newtio.c_cflag = B115200 | CS8 | CLOCAL | CREAD;
	newtio.c_iflag = IGNPAR;
	newtio.c_oflag = 0;
	newtio.c_lflag = 0;
	newtio.c_cc[VTIME] = 128;    // time-out 값으로 사용된다. time-out 값은 TIME*0.1초 이다.
	newtio.c_cc[VMIN] = 0;     // MIN은 read가 리턴되기 위한 최소한의 문자 개수

	tcflush(handle, TCIFLUSH);
	tcsetattr(handle, TCSANOW, &newtio);
sleep(5);
    // RPLIDAR A1과 통신을 위한 장치 드라이버 생성.
    // RPLIDAR 제어는 드라이버를 통해서 진행된다: 예. rplidarA1 -> functionName().
    std::cout << "[INFO] RPLIDAR DRIVER:";
    RPlidarDriver* rplidarDRIVER = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
    if (!rplidarDRIVER) {   // EXIT UPON ERROR
        std::cout << " ...FAILED!";
        std::cout << "[ERROR] FAILED TO CREATE DRIVER." << std::endl;
        exit(1);
    }
    std::cout << " ...READY!" << std::endl;

    // 시리얼 포트 경로 "/dev/ttyUSB0"를 통해
    /*
        >> `rp::standalone::rplidar::connet()`: RPLidar 드라이버를 연결할 RPLIDAR A1 장치와 어떤 시리얼 포트를 사용할 것인지,
            그리고 통신채널에서 송수률(baud rate)인 초당 최대 비트, 즉 bit/sec을 선택한다. 일반적으로 RPLIDAR 모델의baud rate는 115200으로 설정한다.
            ...만일 드라이버와 장치의 연결이 성공되었으면 숫자 0을 반환한다.
    */
    std::cout << "[INFO] CONNECTION:";
    u_result RESULT = rplidarDRIVER->connect("/dev/ttyUSB0", 115200);
    if(IS_FAIL(RESULT)) {  // 연결이 실패하였으면 에러를 알리고, 객체를 자동적으로 파괴한다.

        std::cout << " ...FAILED! RETRYING:";
        system("sudo -S chmod 0666 /dev/ttyUSB0");
        if (IS_FAIL(RESULT = rplidarDRIVER->connect("/dev/ttyUSB0", 115200))){
            std::cout << " ...FAILED!" << std::endl << "[ERROR] FAILED TO CONNECT TO LIDAR: " << "0x" << std::hex << RESULT << std::dec << std::endl;
            exit(1);
        }
    }


    // RPLIDAR에는 여러 종류의 스캔 모드가 있는데, 이 중에서 일반 스캔 모드를 실행한다.
    /*
        >> `rp::standalone::rplidar::startScanExpress(<force>,<use_TypicalScan>,<options>,<outUsedScanMode>)`:
            ...<force>           - 모터 작동 여부를 떠나 가ㅇ제(force)로 스캔 결과를 반환하도록 한다.
            ...<use_TypicalScan> - true는 일반 스캔모드(초당 8k 샘플링), false는 호환용 스캔모드(초당 2k 샘플링).
            ...<options>         - 0을 사용하도록 권장하며, 그 이외의 설명은 없다.
            ...<outUsedScanMode> - RPLIDAR가 사용할 스캔모드 가ㅄ이 반환되는 변수.
    */
    std::cout << " ...SUCCESS!" << std::endl;
    std::cout << "[INFO] MOTOR START:";
    rplidarDRIVER -> startMotor();
    std::cout << " ...SUCCESS!" << std::endl;

    RplidarScanMode scanMode;
    rplidarDRIVER -> startScan(false, true, 0, &scanMode);

    size_t nodeCount; rplidar_response_measurement_node_hq_t nodes[8192];
    int rplidarDISTANCES[4] = {-1}; int avoid = 0; int slide = 0;
    signal(SIGINT, ctrlc);

    std::thread th_read (fifo_from_read);
    // std::thread th_scan (rplidarSCAN, rplidarDRIVER, RESULT, rplidarDISTANCES, nodes, nodeCount);


    th_read.detach();

    while(1){
	if (isEnd) break;
        if (rplidarSTOP) break;
        
        // 노드 개수(8192)를 계산적으로 구한다.
        nodeCount = sizeof(nodes)/sizeof(rplidar_response_measurement_node_hq_t);
        int rplidarDISTANCE[4] = {-1};

        // 완전한 0-360도, 즉 한 사이클의 스캔이 완료되었으면 스캔 정보를 획득한다.
        /*
            >> `grabScanDataHq(<nodebuffer>,<count>)`: 본 API로 획득한 정보들은 항상 다음과 같은 특징을 가진다:

                1) 획득한 데이터 행렬의 첫 번째 노드, 즉 <nodebuffer>[0]는 첫 번째 스캔 샘플가ㅄ이다 (start_bit == 1).
                2) 데이터 전체는 정확히 한 번의 360도 사이클에 대한 스캔 정보만을 지니고 있으며, 그 이상 혹은 그 이하도 아니다.
                3) 각도 정보는 항상 오름차순으로 나열되어 있지 않다. 이는 ascendScanData API를 사용하여 오름차순으로 재배열 가능하다.

                ...<nodebuffer> - API가 스캔 정보를 저장할 수 있는 버퍼.
                ...<count>      - API가 버퍼에게 전달할 수 있는 최대 데이터 개수를 초기설정해야 한다.
                                API의 동작이 끝났으면 해당 파라미터로 입력된 변수는 실제로 스캔된 정보 개수가 할당된다 (예. 8192 -> 545)
        */
        RESULT = rplidarDRIVER->grabScanDataHq(nodes, nodeCount);
        if (IS_OK(RESULT)) {    // 스캔을 성공하였을 경우 순서를 오름차순으로 재배열한다.
            
            rplidarDRIVER -> ascendScanData(nodes, nodeCount);
        }
        else if (IS_FAIL(RESULT)){  // 스캔을 실패하였을 경우 아래의 코드를 실행한다.
            std::cout << "[ERROR] FAILED TO SCAN USING LIDAR: " << "0x" << std::hex << RESULT << std::dec << std::endl;
            continue;
        }
        
        rplidarMEASURE(rplidarDISTANCE, nodes, nodeCount);
        move = rplidarMOVE(rplidarDISTANCE, buff[0], &avoid, &slide);
	buff_a[0] = move;
        rplidarRESULT(rplidarDISTANCE, move);
	write(handle, buff_a, 1);
	tcflush(handle, TCIFLUSH);

    }
close(handle);// END LOOP: WHILE(1)


    // RPLIDAR 드라이버가 존재할 경우...
    if (rplidarDRIVER){

        // RPLIDAR가 정상적으로 작동한 경우...
        if (IS_OK(RESULT)){
            // RPLIDAR A1 센서의 모터 동작을 중지.
            std::cout << "[INFO] STOP MOTOR:";
            rplidarDRIVER -> stopMotor();
            std::cout << " ...SUCCESS!" << std::endl;

            // RPLIDAR A1 센서와 장치 드라이버 통신 단절.
            std::cout << "[INFO] DISCONNECTING:";
            rplidarDRIVER -> disconnect();
            std::cout << " ...SUCCESS!" << std::endl;
        }

        // RPLIDAR A1과 통신을 위한 장치 드라이버 제거.
        std::cout << "[INFO] CLOSING DRIVER:";
        RPlidarDriver::DisposeDriver(rplidarDRIVER);
        std::cout << " ...SUCCESS!" << std::endl;
    }

    rplidarDRIVER = NULL;


    return 0;
}





void rplidarMEASURE(int rplidarDISTANCES[4], rplidar_response_measurement_node_hq_t nodes[8192], size_t nodeCount){

    // RETRIEVE THE SCANNED DATA ONE-BY-ONE.
    for (int i = 0; i < nodeCount; i++){    // START OF FOR LOOP: READING SCAN DATA

        // ANGLE (DEGREE) AND DISTANCE (MILLIMETER): DISTANCE OUTSIDE THE RANGE IS NOTED 0.
        float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
        int distance = nodes[i].dist_mm_q2 / (1 << 2);

        // std::cout << nodes[i].angle_z_q14 * 90.f / (1 << 14) << ", " << nodes[i].dist_mm_q2 / (1 << 2) << std::endl;

        // DIRECTION: FRONT (112)
        if (angle >= 124 && angle <= 236) {
            if (rplidarDISTANCES[F] <= 0 || distance == 0) rplidarDISTANCES[F] = std::max(rplidarDISTANCES[F],distance);
            else rplidarDISTANCES[F] = std::min(rplidarDISTANCES[F], distance);
        }

        // DIRECTION: BACK (60)
        else if ((angle <=  30 || angle >= 330) && distance > 300) {
            if (rplidarDISTANCES[B] <= 0 || distance == 0) rplidarDISTANCES[B] = std::max(rplidarDISTANCES[B],distance);
            else rplidarDISTANCES[B] = std::min(rplidarDISTANCES[B], distance);
        }

        // DIRECTION: LEFT (94)
        else if (angle > 30 & angle < 124) {
            if (rplidarDISTANCES[L] <= 0 || distance == 0) rplidarDISTANCES[L] = std::max(rplidarDISTANCES[L],distance);
            else rplidarDISTANCES[L] = std::min(rplidarDISTANCES[L], distance);
        }

        // DIRECTION: RIGHT (94)
        else if (angle > 236 && angle < 330) {
            if (rplidarDISTANCES[R] <= 0 || distance == 0) rplidarDISTANCES[R] = std::max(rplidarDISTANCES[R],distance);
            else rplidarDISTANCES[R] = std::min(rplidarDISTANCES[R], distance);
        }

        // REDUNDANT DIRECTION
        else continue;

    }// END LOOP: FOR(COMPRESS)


}





// RPLIDAR 거리와 플랫폼 이동 신호를 통합하여 우선순위를 결정한다.
char rplidarMOVE(int rplidarDISTANCES[4],char move, int* avoid, int* slide){

    /* SIGNAL STOP (CURRENTLY NOT AVOIDING) */
    if (move == STOP && (*avoid != 0 || *slide != 0) ) {
        *avoid = 0; *slide = 0;
        return STOP;
    }

    /* SIGNAL GO (WHEN MEASURED) */
    else if (move == GO && !(rplidarDISTANCES[F] < 0) && *avoid != 1) {
        
        // RESET AVOIDIND PARAMETER
        *slide = 0;

        // I. OBSTACLE FOUND IN RANGE
        if (0 < rplidarDISTANCES[F] && rplidarDISTANCES[F] <= DIST_STOP) {

            // (1) BACKING UP
            if (*avoid == -1) {
                goto backing;
            }

            // (2) AVOID LEFT 
            if (  (rplidarDISTANCES[L] > DIST_REF && rplidarDISTANCES[L] >= rplidarDISTANCES[R] && rplidarDISTANCES[R] > 0) || rplidarDISTANCES[L] == 0 ){
                *avoid = 1; *slide = L;
                return LEFT;
            }
            // (3) AVOID RIGHT
            else if (  (rplidarDISTANCES[R] > DIST_REF  && rplidarDISTANCES[L] <= rplidarDISTANCES[R] &&  rplidarDISTANCES[L] > 0) || rplidarDISTANCES[R] == 0 ){
                *avoid = 1; *slide = R;
                return RIGHT;
            }
            
            // (4) AVOID BACK
        backing:
            if ( rplidarDISTANCES[B] == 0 || rplidarDISTANCES[B] > DIST_REF){
                *avoid = -1; *slide = 0;
                return BACK;
            }
            // (5) STOP IF NO OPTION
            else {
                *avoid = 0; *slide = 0;
                return STOP;
            }
        }
        
        // II. OBSTACLE NOT FOUND IN RANGE
        *avoid = 0; *slide = 0;
        return GO;
    }

    /* MOVING LEFT OR AVOIDING LEFT (WHEN MEASURED) */
    else if ( (move == LEFT || *slide == L ) && !(rplidarDISTANCES[L] < 0)) {

        // MOVING LEFT
        if (*avoid == 0 && *slide == 0){
            return LEFT;
        }

        // AVOIDING LEFT 
        else if (*avoid == 1 && *slide == L  ){

            // BUT OBSTACLE STILL FOUND AT FRONT
            if (0 < rplidarDISTANCES[F] && rplidarDISTANCES[F] <= DIST_STOP){
                // OBSTACLE AT LEFT
                if (rplidarDISTANCES[L] < DIST_REF) return STOP;
                // IF NOT
                else return LEFT;
            }
            // NO MORE BLOCKING THE WAY
            else {
                *avoid = 0; *slide = L;
                return LEFT;//
            }
        }

        // AFTER AVOIDING, PASSING OBSTACLE LEFT SIDE
        else if (*avoid == 0 && *slide == L) {

            // IF OBSTACLE EXIST ON RIGHT SIDE
            if (0 < rplidarDISTANCES[R] && rplidarDISTANCES[R] <= DIST_REF) {
                // AND IF OBSTACLE IS BLOCKING FRONT
                if (0 < rplidarDISTANCES[F] && rplidarDISTANCES[F] <= DIST_STOP) return STOP;
                // BUT IF NOT
                else return GO;
            }
            // WHEN COMPLETELY PASSED
            else {
                *avoid = 0; *slide = 0;
                return STOP;//
            }
        }

    }

    /* MOVING RIGHT OR AVOIDING RIGHT (WHEN MEASURED) */
    else if ( (move == RIGHT || *slide == R ) && !(rplidarDISTANCES[R] < 0)) {

        // MOVING RIGHT
        if (*avoid == 0 && *slide == 0){
            return RIGHT;
        }

        // AVOIDING RIGHT 
        else if (*avoid == 1 && *slide == R  ){

            // BUT OBSTACLE STILL FOUND AT FRONT
            if (0 < rplidarDISTANCES[F] && rplidarDISTANCES[F] <= DIST_STOP){
                // OBSTACLE AT RIGHT
                if (rplidarDISTANCES[R] < DIST_REF) return STOP;
                // IF NOT
                else return RIGHT;
            }
            // NO MORE BLOCKING THE WAY
            else {
                *avoid = 0; *slide = R;
                return STOP;
            }
        }

        // AFTER AVOIDING, PASSING OBSTACLE RIGHT SIDE
        else if (*avoid == 0 && *slide == R) {

            // IF OBSTACLE EXIST ON LEFT SIDE
            if (0 < rplidarDISTANCES[R] && rplidarDISTANCES[R] <= DIST_REF){
                // AND IF OBSTACLE IS BLOCKING FRONT
                if (0 < rplidarDISTANCES[F] && rplidarDISTANCES[F] <= DIST_STOP) return STOP;
                // BUT IF NOT
                else return GO;
            } 
            // WHEN COMPLETELY PASSED
            else {
                *avoid = 0; *slide = 0;
                return STOP;
            }
        }
    }

    /* BACK (WHEN MEASURED) */
    else if (move == BACK && !(rplidarDISTANCES[B] < 0)) {

        // SIGNAL BACK BUT OBSTACLE FOUND
        if (rplidarDISTANCES[B] > 0 && rplidarDISTANCES[B] <= DIST_REF) return STOP;
        // IF NOT
        return BACK;
    } 

    else {
        *avoid = 0; *slide = 0;
        return STOP;
    }

}



void rplidarRESULT(int rplidarDISTANCES[4], char move) {

    std::cout << "=============" << std::endl;
    std::cout << "| F: " << rplidarDISTANCES[F] << std::endl;
    std::cout << "| B: " << rplidarDISTANCES[B] << std::endl;
    std::cout << "| L: " << rplidarDISTANCES[L] << std::endl;
    std::cout << "| R: " << rplidarDISTANCES[R] << std::endl;
    std::cout << "+--- INPUT: " << buff[0] << std::endl;
    std::cout << "+--- MOVE: " << move << std::endl;
    
}
