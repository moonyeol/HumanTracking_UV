// RPLIDAR A1 센서를 C++으로 사용하는 방법: https://github.com/Slamtec/rplidar_sdk

#include <iostream>
#include <rplidar.h>
#include <cmath>        // 용도: abs()와 round() 및 lround() 함수 활용.

#define CYCLE 360
#define DIRECTION 4

using namespace rp::standalone::rplidar;
using namespace std;

int main() {

    // RPLIDAR A1과 통신을 위한 장치 드라이버 생성.
    // RPLIDAR 제어는 드라이버를 통해서 진행된다: 예. rplidarA1 -> functionName().
    rp::standalone::rplidar::RPlidarDriver * rplidarA1 = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
    int distances[DIRECTION]={0};

    // 시리얼 포트 경로 "/dev/ttyUSB0"를 통해
    /*
        >> `rp::standalone::rplidar::connet()`: RPLidar 드라이버를 연결할 RPLIDAR A1 장치와 어떤 시리얼 포트를 사용할 것인지,
            그리고 통신채널에서 송수률(baud rate)인 초당 최대 비트, 즉 bit/sec을 선택한다. 일반적으로 RPLIDAR 모델의baud rate는 115200으로 설정한다.
            ...만일 드라이버와 장치의 연결이 성공되었으면 숫자 0을 반환한다.
    */
    u_result result = rplidarA1->connect("/dev/ttyUSB0", 115200);

    // 연결이 성공하였으면 아래의 코드를 실행한다
    // ...res = 0이면 연결에 성공한 것이다.
    if (IS_OK(result))
    {
        
        // RPLIDAR 모터 동작.
        rplidarA1 -> startMotor();

        // RPLIDAR에는 여러 종류의 스캔 모드가 있는데, 이 중에서 일반 스캔 모드를 실행한다.
        /*
            >> `rp::standalone::rplidar::startScanExpress(<force>,<use_TypicalScan>,<options>,<outUsedScanMode>)`:
                ...<force>           - 모터 작동 여부를 떠나 가ㅇ제(force)로 스캔 결과를 반환하도록 한다.
                ...<use_TypicalScan> - true는 일반 스캔모드(초당 8k 샘플링), false는 호환용 스캔모드(초당 2k 샘플링).
                ...<options>         - 0을 사용하도록 권장하며, 그 이외의 설명은 없다.
                ...<outUsedScanMode> - RPLIDAR가 사용할 스캔모드 값이 반환되는 변수.
        */
        RplidarScanMode scanMode;
        rplidarA1 -> startScan(false, true, 0, &scanMode);


        while (true) {  // START OF WHILE LOOP: INDEFINITE SCANNING CYCLE

            // 스캔 데이터인 노드(node)를 담을 수 있는 배열을 생성한다.
            rplidar_response_measurement_node_hq_t nodes[8192];

            // 노드 개수(8192)를 계산적으로 구한다.
            size_t nodeCount = sizeof(nodes)/sizeof(rplidar_response_measurement_node_hq_t);

            // 완전한 0-360도, 즉 한 사이클의 스캔이 완료되었으면 스캔 정보를 획득한다.
            /*
                >> `grabScanDataHq(<nodebuffer>,<count>)`: 본 API로 획득한 정보들은 항상 다음과 같은 특징을 가진다:

                    1) 획득한 데이터 행렬의 첫 번째 노드, 즉 <nodebuffer>[0]는 첫 번째 스캔 샘플값이다 (start_bit == 1).
                    2) 데이터 전체는 정확히 한 번의 360도 사이클에 대한 스캔 정보만을 지니고 있으며, 그 이상 혹은 그 이하도 아니다.
                    3) 각도 정보는 항상 오름차순으로 나열되어 있지 않다. 이는 ascendScanData API를 사용하여 오름차순으로 재배열 가능하다.

                    ...<nodebuffer> - API가 스캔 정보를 저장할 수 있는 버퍼.
                    ...<count>      - API가 버퍼에게 전달할 수 있는 최대 데이터 개수를 초기설정해야 한다.
                                    API의 동작이 끝났으면 해당 파라미터로 입력된 변수는 실제로 스캔된 정보 개수가 할당된다 (예. 8192 -> 545)
            */
            result = rplidarA1->grabScanDataHq(nodes, nodeCount);
        

            // 스캔을 성공하였을 경우 아래의 코드를 실행한다.
            if (IS_OK(result)) {    // START OF IF CONDITION: IF SCAN IS COMPLETE

                // <angleRange>: 총 방향 개수, <distances[]>: 거리를 담는 배열, <count>: 방향 카운터, <angleOF_prev>: 이전 위상값을 받아내기 위한 변수.
                int angleRange = CYCLE/DIRECTION;
                int count = 0, angleOFF_prev = NULL;

                // 거리값을 계산하고 정리하기 위해 사용되는 임시 저장변수.
                int distancesTEMP[DIRECTION] = {0};

                // 순서를 오름차순으로 재배열한다.
                rplidarA1 -> ascendScanData(nodes, nodeCount);

                // 스캔 결과를 오름차순으로 하나씩 확인한다.
                for (int i = 0; i < nodeCount; i++){    // START OF FOR LOOP: READING SCAN DATA
                    

                    // 각도는 도 단위 (+ 위상), 거리는 밀리미터 단위로 선정 (범위외 거리는 0으로 반환).
                    float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
                    float distance = nodes[i].dist_mm_q2 / (1 << 2);

                    // 위상 추가하여 방향성 교정.
                    angle = angle + angleRange/2;

                    // 하나의 방향이라고 인지할 수 있도록 정해놓은 batch 범위가 있으며, 중앙에서 얼마나 벗어난 각도인지 확인.
                    // 값이 크면 클수록 중앙과 가깝다는 의미.
                    int angleOFF = lround(angle) % angleRange;
                    angleOFF = abs(angleOFF - angleRange/2);
                    
                    // 현재 위상값이 0이고 이전 위상값이 1이면 방향 카운터를 증가시킨다.
                    // 반대로 설정하면 초반에 바로 (현재 = 1, 이전 = 0) 값이 나올 수 있어 오류는 발생하지 않지만 첫 방향의 최소거리가 계산되지 않는다.
                    if (angleOFF == 0 && angleOFF_prev == 1) count++;
                    
                    // 처리되지 않은 나머지 전방 각도 당 거리를 계산하기 위하여 방향 카운터 초기화.
                    if (count == DIRECTION) count = 0;

                    // 루프를 돌기 전에 현재 위상값을 이전 위상값으로 할당한다.
                    angleOFF_prev = angleOFF;

                    // 최소거리를 저장한다.
                    if (distancesTEMP[count] == 0) distancesTEMP[count] = distance;
                    else if (distancesTEMP[count] > distance && distance != 0) distancesTEMP[count] = distance;

                }   // END OF FOR LOOP: READING SCAN DATA.

                for (int i = 0; i < DIRECTION; i++) distances[i]  = distancesTEMP[i];

            }   // END OF IF CONDITION: IF SCAN IS COMPLETE
            
            // FOR A SINGLE CYCLE
            // break;

            // 스캔을 실패하였을 경우 아래의 코드를 실행한다.
            if (IS_FAIL(result))
            {   
                std::cout << "[ERROR] FAILED TO SCAN USING LIDAR." << std::endl;
                break;
            }

            for (int i = 0; i < DIRECTION; i++){
                cout << distances[i] << endl;
            }
            cout << "====" << endl;

        }   // END OF WHILE LOOP: INDEFINITE SCANNING CYCLE


        // RPLIDAR 모터 중지.
        rplidarA1 -> stopMotor();
        
        // 드라이버의 장치 연결을 끊는다.
        rplidarA1 -> disconnect();
    }

    // 연결이 실패하였으면 아래의 코드를 실행한다.
    else {fprintf(stderr, "Failed to connect to LIDAR %08x\r\n", result);}

    // RPLIDAR A1과 통신을 위한 장치 드라이버 제거.
    RPlidarDriver::DisposeDriver(rplidarA1);

    return 0;

}
