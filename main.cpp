#include <pybind11/embed.h>

namespace py = pybind11
PSSI_HandleTypeDef hpssi;

void setupPSSI()
{
    hpssi.Instance = PSSI;

    hpssi.Init.BusWidth = HAL_PSSI_8BITS;
    hpssi.Init.DataEnablePolarity = HAL_PSSI_DEPOLARITY_HIGH;
    hpssi.Init.ReadyPolarity = HAL_PSSI_RDYPOLARITY_HIGH;
    hpssi.Init.ClockPolarity = HAL_PSSI_FALLING_EDGE;

    HAL_PSSI_Init(&hpssi);
}

void calibrate_cams(){
    py::module_ CC = py::module_::import("Calc_coords");
    py::tuple calibration =
        CC.attr("calibrate")(
            left_image,
            right_image
        );

    vision.attr("save_calibration")(
        "calibration.npz",
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        Q
    );

    py::array map_left_x =
        calibration[0].cast<py::array>();

    py::array map_left_y =
        calibration[1].cast<py::array>();

    py::array map_right_x =
        calibration[2].cast<py::array>();

    py::array map_right_y =
        calibration[3].cast<py::array>();

    py::array Q =
        calibration[4].cast<py::array>();
    
}

uint8_t frameBuffer[WIDTH * HEIGHT];

HAL_PSSI_Receive_DMA(
    &hpssi,
    frameBuffer,
    sizeof(frameBuffer)
);

void main() {
    py::tuple calibration =
        vision.attr("load_calibration")(
            "calibration.npz"
        );

    py::array map_left_x = calibration[0].cast<py::array>();
    py::array map_left_y = calibration[1].cast<py::array>();
    py::array map_right_x = calibration[2].cast<py::array>();
    py::array map_right_y = calibration[3].cast<py::array>();
    py::array Q = calibration[4].cast<py::array>();

    py::scoped_interpreter guard{};
    
    py::module_ CC = py::module_::import("Calc_coords");
    py::module_ SSLAM = py::module_::import("SSLAM");
    
    //Placeholder values
    int left_img = 0;
    int right_img = 1;

    py::tuple rectified = CC.attr("rectify")(
        left_img,
        right_img,
        map_left_x, 
        map_left_y,
        map_right_x,
        map_right_y
    )



}

