#include <iostream>
#include <rplidar.h>    // RPLIDAR SDK 접속
#include <algorithm>    // 수학적 거리 계산 전용
#include "rplidar.hpp"

#define CYCLE 360
#define DIRECTION 4

#define DIST_STOP 300   // 장애물 기준 거리를 500mm, 즉 0.5미터로 잡는다.
#define DIST_REF 500    // 방향을 틀었을 때, 최소한 0.7미터의 여유가 있을 때로 선택한다.

#define GO "g"
#define BACK "b"
#define LEFT "l"
#define RIGHT "r"
#define STOP "s"

using namespace rp::standalone::rplidar;

/*__________ START: CONSTRUCTOR AND DESTRUCTOR __________*/
// CONSTRUCTOR
rplidar::rplidar(): RESULT(NULL), rplidarDRIVER(NULL), platformMOVE(NULL), avoid(false)
{

    // RPLIDAR A1과 통신을 위한 장치 드라이버 생성.
    // RPLIDAR 제어는 드라이버를 통해서 진행된다: 예. rplidarA1 -> functionName().
    std::cout << "[INFO] RPLIDAR DRIVER:";
    rplidarDRIVER = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
    if (!rplidarDRIVER) {
        std::cout << "...FAILED!";
        std::cout << "[ERROR] FAILED TO CREATE DRIVER." << std::endl;
        
    }
    std::cout << " ...READY!" << std::endl;
    // 시리얼 포트 경로 "/dev/ttyUSB0"를 통해
    /*
        >> `rp::standalone::rplidar::connet()`: RPLidar 드라이버를 연결할 RPLIDAR A1 장치와 어떤 시리얼 포트를 사용할 것인지,
            그리고 통신채널에서 송수률(baud rate)인 초당 최대 비트, 즉 bit/sec을 선택한다. 일반적으로 RPLIDAR 모델의baud rate는 115200으로 설정한다.
            ...만일 드라이버와 장치의 연결이 성공되었으면 숫자 0을 반환한다.
    */
    std::cout << "[INFO] CONNECTION:";
    RESULT = rplidarDRIVER->connect("/dev/ttyUSB0", 115200);

    // 연결이 성공하였으면 아래의 코드를 실행한다
    if(IS_OK(RESULT)) {
        // RPLIDAR 모터 동작.
        std::cout << " ...SUCCESS!" << std::endl;
        std::cout << "[INFO] MOTOR START:";
        rplidarDRIVER -> startMotor();
        std::cout << " ...SUCCESS!" << std::endl;

        // RPLIDAR에는 여러 종류의 스캔 모드가 있는데, 이 중에서 일반 스캔 모드를 실행한다.
        /*
            >> `rp::standalone::rplidar::startScanExpress(<force>,<use_TypicalScan>,<options>,<outUsedScanMode>)`:
                ...<force>           - 모터 작동 여부를 떠나 가ㅇ제(force)로 스캔 결과를 반환하도록 한다.
                ...<use_TypicalScan> - true는 일반 스캔모드(초당 8k 샘플링), false는 호환용 스캔모드(초당 2k 샘플링).
                ...<options>         - 0을 사용하도록 권장하며, 그 이외의 설명은 없다.
                ...<outUsedScanMode> - RPLIDAR가 사용할 스캔모드 가ㅄ이 반환되는 변수.
        */
        RplidarScanMode scanMode;
        rplidarDRIVER -> startScan(false, true, 0, &scanMode);
    }

    // 연결이 실패하였으면 에러를 알리고, 객체를 자동적으로 파괴한다.
    else {
        std::cout << "...FAILED!" << std::endl;
        std::cout << "[ERROR] FAILED TO CONNECT TO LIDAR: " << "0x" << std::hex << RESULT << std::dec << std::endl;
        this->~rplidar();
    }
}

// DESTRUCTOR
rplidar::~rplidar(){

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

}
/*__________ END: CONSTRUCTOR AND DESTRUCTOR __________*/



/*__________ START: PUBLIC MEMBERS __________*/
// RPLIDAR A1 센서 스캔 결과를 가져온다.
void rplidar::scan(){

    // 노드 개수(8192)를 계산적으로 구한다.
    this->nodeCount = sizeof(this->nodes)/sizeof(rplidar_response_measurement_node_hq_t);

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
    RESULT = rplidarDRIVER->grabScanDataHq(this->nodes, this->nodeCount);

    // 스캔을 성공하였을 경우 아래의 코드를 실행한다.
    if (IS_OK(RESULT)) {

        // 순서를 오름차순으로 재배열한다.
        rplidarDRIVER -> ascendScanData(this->nodes, this->nodeCount);
        this->compressDistance();
    }

    // 스캔을 실패하였을 경우 아래의 코드를 실행한다.
    else if (IS_FAIL(this->RESULT))
    {   
        std::cout << "[ERROR] FAILED TO SCAN USING LIDAR: " << "0x" << std::hex << RESULT << std::dec << std::endl;
    }
}

// 우선순위 결정 후 최종적으로 보내줄 이동신호를 반환한다.
char* rplidar::move(char* MOVE){
    platformMOVE = this->behavior(MOVE);
    return platformMOVE;
}

// 계산된 거리와 최종 이동방향을 보여준다.
void rplidar::result(){
    
    std::cout << "=============" << std::endl;
    std::cout << "F: " << this->rplidarDIST[2] << std::endl;
    std::cout << "B: " << this->rplidarDIST[0] << std::endl;
    std::cout << "L: " << this->rplidarDIST[1] << std::endl;
    std::cout << "R: " << this->rplidarDIST[3] << std::endl;
    std::cout << "DIRECTION: " << this->platformMOVE << std::endl;
    
}
/*__________ END: PUBLIC MEMBERS __________*/



/*__________ START: PRIVATE MEMBERS __________*/
// RPLIDAR A1 센서 스캔 결과를 통해 사방 거리를 하나로 축약한다.
void rplidar::compressDistance(){

    std::fill(rplidarDIST, rplidarDIST+DIRECTION, -1);

    // RETRIEVE THE SCANNED DATA ONE-BY-ONE.
    for (int i = 0; i < nodeCount; i++){    // START OF FOR LOOP: READING SCAN DATA

        // ANGLE (DEGREE) AND DISTANCE (MILLIMETER): DISTANCE OUTSIDE THE RANGE IS NOTED 0.
        float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
        float distance = nodes[i].dist_mm_q2 / (1 << 2);

        // std::cout << nodes[i].angle_z_q14 * 90.f / (1 << 14) << ", " << nodes[i].dist_mm_q2 / (1 << 2) << std::endl;

        // DIRECTION: FRONT (112)
        if (angle >= 124 && angle <= 236) {
            if (rplidarDIST[2] <= 0 || distance == 0) rplidarDIST[2] = std::max(rplidarDIST[2],distance);
            else rplidarDIST[2] = std::min(rplidarDIST[2], distance);
        }

        // DIRECTION: BACK (60)
        else if ((angle <=  30 || angle >= 330) && distance > 300) {
            if (rplidarDIST[0] <= 0 || distance == 0) rplidarDIST[0] = std::max(rplidarDIST[0],distance);
            else rplidarDIST[0] = std::min(rplidarDIST[0], distance);
        }

        // DIRECTION: LEFT (94)
        else if (angle > 30 & angle < 124) {
            if (rplidarDIST[1] <= 0 || distance == 0) rplidarDIST[1] = std::max(rplidarDIST[1],distance);
            else rplidarDIST[1] = std::min(rplidarDIST[1], distance);
        }

        // DIRECTION: RIGHT (94)
        else if (angle > 236 && angle < 330) {
            if (rplidarDIST[3] <= 0 || distance == 0) rplidarDIST[3] = std::max(rplidarDIST[3],distance);
            else rplidarDIST[3] = std::min(rplidarDIST[3], distance);
        }

        // REDUNDANT DIRECTION
        else continue;

    }
}

// RPLIDAR 거리와 플랫폼 이동 신호를 통합하여 우선순위를 결정한다.
char* rplidar::behavior(char* move){

    // STOP
    if (*move == *STOP) {
        avoid = 0;
        return STOP;
    }
    // GO
    else if (*move == *GO && !(rplidarDIST[2] < 0)) {
        
        // OBSTACLE FOUND IN RANGE
        if (0 < rplidarDIST[2] && rplidarDIST[2] <= DIST_STOP){

            // BACKING UP
            if (avoid == -1) {
                goto backing;
            }
            
            // AVOID TO LEFT
            if (  ((rplidarDIST[1] > DIST_REF && rplidarDIST[1] >= rplidarDIST[3] && rplidarDIST[3] > 0 ) || rplidarDIST[1] == 0) ){
                avoid = 1;
                return LEFT;
            }
            // AVOID TO RIGHT
            else if (  ((rplidarDIST[3] > DIST_REF  && rplidarDIST[1] <= rplidarDIST[3] &&  rplidarDIST[1] > 0 ) || rplidarDIST[3] == 0 ) ){
                avoid = 1;
                return RIGHT;
            }
            // AVOID BACK
            else if ( rplidarDIST[0] == 0 || rplidarDIST[0] > DIST_REF){
            backing:
                avoid = -1;
                return BACK;
            }
            else
                return STOP;
        }
        
        // OBSTACLE NOT FOUND IN RANGE
        avoid = 0;
        return GO;
    }

    // LEFT
    else if (*move == *LEFT && !(rplidarDIST[1] < 0)) {

        // AVOIDING RIGHT BUT OBSTACLE STILL FOUND
        if (avoid && (0 < rplidarDIST[2] && rplidarDIST[2] <= DIST_STOP) ){
            // OBSTACLE AT RIGHT
            if (rplidarDIST[3] < DIST_REF) return STOP;
            // IF NOT
            return RIGHT;
        }

        avoid = 0;
        return LEFT;
    }

    // RIGHT
    else if (*move == *RIGHT && !(rplidarDIST[3] < 0)) {

        // AVOIDING LEFT BUT OBSTACLE STILL FOUND
        if (avoid && (0 < rplidarDIST[2] && rplidarDIST[2] <= DIST_STOP) ){
            // OBSTACLE AT LEFT
            if (rplidarDIST[1] < DIST_REF) return STOP;
            // IF NOT
            return LEFT;
        }

        avoid = 0;
        return RIGHT;
    }

    // BACK
    else if (*move == *BACK && !(rplidarDIST[0] < 0)) {
        if (rplidarDIST[0] > 0 && rplidarDIST[0] <= DIST_REF) return STOP;
        return BACK;
    } 

    else {
        avoid = 0;
        return STOP;
    }

}
/*__________ END: PRIVATE MEMBERS __________*/
