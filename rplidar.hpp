#ifndef RPLIDAR_HPP
#define RPLIDAR_HPP

#define DIRECTION 4

using namespace rp::standalone::rplidar;

class rplidar{

    public:
        rplidar();
        ~rplidar();

        // RPLIDAT A1 센서로 한 사이클 스캔한다.
        void scan();
    
        // RPLIDAR A1 센서 스캔 결과를 가져온다.
        void retrieve();

        // 우선순위 결정 후 최종적으로 보내줄 이동신호를 반환한다.
        char* returnMove(char*);

        // 계산된 거리와 최종 이동방향을 보여준다.
        void result();

    private:
        RPlidarDriver* rplidarDRIVER;

        // RPLIDAR A1에서 측정한 거리와 플랫폼 이동신호를 담는 변수이다.
        int rplidarDIST[DIRECTION] = {0};
        char* platformMOVE;

        // RPLIDAR A1 센서와 드라이버와의 통신 결과를 담는다.
        /*
            >> Reference: http://docs.ros.org/kinetic/api/rplidar_ros/html/rptypes_8h_source.html (LINE: 100)
            >> type defined: uint32_t
            >> OK:                   0 ->   0.  0.  0.  0
            >> FAIL_BIT:    0x80000000 -> 128.  0.  0.  0
             - INVALID DATA:   |0x8000 -> 128.  0.128.  0
             - OPERATION FAIL: |0x8001 -> 128.  0.128.  1
             - TIMEOUT         |0x8002 -> 128.  0.128.  2
             - STOPPED         |0x8003 -> 128.  0.128.  3
             - NOT SUPPORTED:  |0x8004 -> 128.  0.128.  4
             - INVALID FORM :  |0x8005 -> 128.  0.128.  5
             - INSUFF MEMORY:  |0x8006 -> 128.  0.128.  6
            >> ALREADY_DONE:      0x20 ->   0.  0.  0. 32
            
            >> IS_OK(x):   BOOL ((x & FAIL_BIT) == 0) [TRUE WHEN x IS 0]
            >> IS_FAIL(x): BOOL  (x & FAIL_BIT)       [TRUE WHEN x IS NOT 0]
        */
        u_result RESULT;

        // 스캔 데이터인 노드(node)를 담을 수 있는 배열을 생성한다.
        rplidar_response_measurement_node_hq_t nodes[8192];
        size_t nodeCount;

        // RPLIDAR A1 센서 스캔 결과를 통해 사방 거리를 하나로 축약한다.
        void compressDistance();

        // RPLIDAR 거리와 플랫폼 이동 신호를 통합하여 우선순위를 결정한다.
        void behavior(char*);
};

#endif
