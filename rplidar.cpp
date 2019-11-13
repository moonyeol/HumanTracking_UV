#include <iostream>
#include <rplidar.h>    // RPLIDAR SDK 접속
#include <cmath>        // 수학적 거리 계산 전용
#include "rplidar.hpp"

#define CYCLE 360
#define DIRECTION 4

#define DIST_STOP 500   // 장애물 기준 거리를 500mm, 즉 0.5미터로 잡는다.
#define DIST_REF 700    // 방향을 틀었을 때, 최소한 0.7미터의 여유가 있을 때로 선택한다.

#define GO "g"
#define BACK "b"
#define LEFT "l"
#define RIGHT "r"
#define STOP "s"

using namespace rp::standalone::rplidar;

/*__________ START: CONSTRUCTOR AND DESTRUCTOR __________*/
// CONSTRUCTOR
rplidar::rplidar(): RESULT(NULL), rplidarDRIVER(NULL), platformMOVE(NULL)
{

    // RPLIDAR A1과 통신을 위한 장치 드라이버 생성.
    // RPLIDAR 제어는 드라이버를 통해서 진행된다: 예. rplidarA1 -> functionName().
    std::cout << "[INFO] RPLIDAR DRIVER:";
    rplidarDRIVER = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
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
    }

    // 연결이 실패하였으면 에러를 알리고, 객체를 자동적으로 파괴한다.
    else {
        std::cout << "...FAILED!" << std::endl;
        std::cout << "[ERROR] FAILED TO CONNECT TO LIDAR." << std::endl;
        this->~rplidar();
    }
}

// DESTRUCTOR
rplidar::~rplidar(){

    // RPLIDAR A1 센서의 모터 동작을 중지.
    std::cout << "[INFO] STOP MOTOR:";
    rplidarDRIVER -> stopMotor();
    std::cout << " ...SUCCESS!" << std::endl;

    // RPLIDAR A1 센서와 장치 드라이버 통신 단절.
    std::cout << "[INFO] DISCONNECTING:";
    rplidarDRIVER -> disconnect();
    std::cout << " ...SUCCESS!" << std::endl;

    // RPLIDAR A1과 통신을 위한 장치 드라이버 제거.
    std::cout << "[INFO] CLOSING DRIVER:";
    RPlidarDriver::DisposeDriver(rplidarDRIVER);
    std::cout << " ...SUCCESS!" << std::endl;
}
/*__________ END: CONSTRUCTOR AND DESTRUCTOR __________*/



/*__________ START: PUBLIC MEMBERS __________*/
// RPLIDAT A1 센서로 한 사이클 스캔한다.
void rplidar::scan() {

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

// RPLIDAR A1 센서 스캔 결과를 가져온다.
void rplidar::retrieve(){

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
        //this->compressDistance();
    }

    // 스캔을 실패하였을 경우 아래의 코드를 실행한다.
    else if (IS_FAIL(this->RESULT))
    {   
        std::cout << "[ERROR] FAILED TO SCAN USING LIDAR." << std::endl;
    }
}

// 우선순위 결정 후 최종적으로 보내줄 이동신호를 반환한다.
char* rplidar::returnMove(char* MOVE){
    return this->behavior(MOVE);
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

    // <angleRange>: 총 방향 개수, <distances[]>: 거리를 담는 배열, <count>: 방향 카운터, <angleOF_prev>: 이전 위상가ㅄ을 받아내기 위한 변수.
    int angleRange = CYCLE/DIRECTION;
    int count = 0, angleOFF_prev = -1;

    std::fill(rplidarDIST, rplidarDIST+DIRECTION, 0);

    // 스캔 결과를 오름차순으로 하나씩 확인한다.
    for (int i = 0; i < nodeCount; i++){    // START OF FOR LOOP: READING SCAN DATA

        // 각도는 도 단위 (+ 위상), 거리는 밀리미터 단위로 선정 (범위외 거리는 0으로 반환).
        float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
        float distance = nodes[i].dist_mm_q2 / (1 << 2);

        // 플랫폼 구조물로 인해서 인식되는 원치않은 각도를 무시한다.
        if ((12.5 < angle && angle < 16.5) || (343.5 < angle && angle <347.5)) continue;

        // std::cout << nodes[i].angle_z_q14 * 90.f / (1 << 14) << ", " << nodes[i].dist_mm_q2 / (1 << 2) << std::endl;

        // 추가위상을 더한다. 이를 modulus 처리하면 가ㅄ이 가장 큰 곳(4방향에서는 거의 45에 가까운 가ㅄ)은 방향성의 중심이 되는 곳이 된다.
        // 반대로 0은 방향성이 바뀌는 경계선이 된다.
        int angleOFF = abs((lround(angle) % angleRange)-angleRange/2);
        
        std::cout << angle - angleRange/2  << ": " << lround(angle) << " % " << angleRange << " = " << angleOFF << std::endl;

        // 현재 위상가ㅄ이 0이고 이전 위상가ㅄ이 1이면 방향 카운터를 증가시킨다.
        // 반대로 설정하면 초반에 바로 (현재 = 1, 이전 = 0) 가ㅄ이 나올 수 있어 오류는 발생하지 않지만 첫 방향의 최소거리가 계산되지 않는다.
        if (angleOFF == 0 && angleOFF_prev == 1) count++;

        // 처리되지 않은 나머지 전방 각도 당 거리를 계산하기 위하여 방향 카운터 초기화.
        if (count == DIRECTION) count = 0;

        // 루프를 돌기 전에 현재 위상가ㅄ을 이전 위상가ㅄ으로 할당한다.
        angleOFF_prev = angleOFF;
        
        // 최소거리를 저장한다.
        if (rplidarDIST[count] == 0) rplidarDIST[count] = distance;
        else if (rplidarDIST[count] > distance && distance != 0) rplidarDIST[count] = distance;
    }

    for(int i =0; i < DIRECTION; i++){

        std::cout << rplidarDIST[i] << std::endl; 

    }
}

// RPLIDAR 거리와 플랫폼 이동 신호를 통합하여 우선순위를 결정한다.
char* rplidar::behavior(char* MOVE){

    this->platformMOVE = MOVE;

    // 정지 신호에는 무조건 정지한다.
    if (*platformMOVE == *STOP) return STOP;

    // 후방 신호가 아닐 경우 장애물이 있으면...
    else if (*platformMOVE != *BACK){

        // ---- 전방에 장애물이 존재할 경우 (0은 측정범위 밖); 후진과 정지는 따로 조건문이 주어져 있으므로 고려하지 않는다.
        if (0 < rplidarDIST[DIRECTION/2] && rplidarDIST[DIRECTION/2] <= DIST_STOP) {

            // 앞에서부터 뒤로 좌우 거리를 확인한다 (후방 제외).
            for (int i = (DIRECTION/2) -1; i > 0; i--){
    
                // 오른쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 왼쪽보다 거리적 여유가 많을 시 오른쪽으로 회전한다.
                if (((rplidarDIST[i] > DIST_REF && rplidarDIST[i] >= rplidarDIST[DIRECTION - i] && rplidarDIST[DIRECTION - i] > 0) || rplidarDIST[i] == 0) && *platformMOVE != *LEFT)
                    {return RIGHT;}
                // 반면 왼쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 오른쪽보다 거리적 여유가 많을 시에는 왼쪽으로 회전한다.
                else if(((rplidarDIST[DIRECTION - i] > DIST_REF  && rplidarDIST[i] <= rplidarDIST[DIRECTION - i] &&  rplidarDIST[i] > 0 ) || rplidarDIST[DIRECTION - i] == 0 ) && *platformMOVE != *RIGHT)
                    {return LEFT;}
            }

            // 후방 거리여부를 확인하고, 전방향이 막혀 있으면 움직이지 않는다.
            if (rplidarDIST[0] > DIST_REF || rplidarDIST[0] == 0) return BACK;
            else return STOP;

        }

        return platformMOVE;
    }
    
    // 후방 신호일 경우 장애물이 있으면 정지.
    else if (*platformMOVE == *BACK && rplidarDIST[0] > 0 && rplidarDIST[0] <= DIST_REF) return STOP;

    return platformMOVE;

}
/*__________ END: PRIVATE MEMBERS __________*/
