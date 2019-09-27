// UGV 프레임을 제작할 때 RPLIDAR A1 센서를 거꾸로 달아버린 탓에, 방향성을 전부 반대로 설정
#include <iostream>
#define DIRECTION 4

#define LEFT "l"
#define RIGHT "r"
#define GO "g"
#define BACK "b"
#define STOP "s"

char* rplidarBehavior(/*char, */char*, int*);

int main() {

    int arr[4] = {250, 600, 700, 500};

    char* direction = rplidarBehavior(GO, arr);

    std::cout << direction << std::endl;

    return 0;
}


char* rplidarBehavior(/*char detectPosition, */char* platformMove, int *distanceRPLIDAR) {

    // REFERENCE
    /*
        >> detectPostion: 영상에 탐지된 대상자가 왼쪽(l) 혹은 오른쪽(r)에 있는지 알려주는 파라미터.
        >> platformMove: 영상에 탐지된 대상자를 기반으로 전진(g), 후진(b), 좌회전(l), 우회전(r), 혹은 정지(s)하는지 알려주는 파라미터.
        >> distanceRPLIDAR = 전방으로 시작으로 시계방향으로 거리를 알려주는 파라미터; {전방, 우, 우X2, ..., 우X(n-1), 후방, 좌X(n-1),  ... , 좌X2, 좌}. 0은 측정범위 밖.
    */

    // 장애물 기준 거리를 300mm, 즉 0.3미터로 잡는다.
    #define DIST_STOP 300

    // 방향을 틀었을 때, 최소한 0.5미터의 여유가 있을 때로 선택한다.
    #define DIST_REF 500

    // 전방에 장애물이 존재할 경우 (0은 측정범위 밖).
    if (0 < *(distanceRPLIDAR + DIRECTION/2) && *(distanceRPLIDAR + DIRECTION/2) <= DIST_STOP){

        // 전 방향의 거리여부를 앞에서부터 뒤로 좌우를 동시에 확인한다 (후방 제외).
        for (int i = (DIRECTION/2) -1; i > 0; i--){

            // 오른쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 왼쪽보다 거리적 여유가 많을 시 오른쪽으로 회전한다.
            if ((*(distanceRPLIDAR + i) > DIST_REF && *(distanceRPLIDAR + i) > *(distanceRPLIDAR + (DIRECTION - i))) || *(distanceRPLIDAR + i) == 0)
                return LEFT;
            // 반면 왼쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 오른쪽보다 거리적 여유가 많을 시에는 왼쪽으로 회전한다.
            else if((*(distanceRPLIDAR + (DIRECTION - i)) > DIST_REF  && *(distanceRPLIDAR + i) < *(distanceRPLIDAR + (DIRECTION - i))) || *(distanceRPLIDAR + (DIRECTION - i)) == 0 )
                return RIGHT;
        }

        // 위의 조건문을 만족하지 않았다는 것은 정해진 기준의 여유보다 거리가 적다는 의미이다.

        // 후방 거리여부를 확인하고, 전방향이 막혀 있으면 움직이지 않는다.
        if (*(distanceRPLIDAR) <= DIST_REF || *(distanceRPLIDAR) == 0) return BACK;
        else return STOP;
    }
    else if (platformMove != LEFT || platformMove != RIGHT) return GO;
    
    return platformMove;
}
