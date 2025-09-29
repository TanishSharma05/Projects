module kernel_bank (
    input  logic        clk,
    input  logic [1:0]  kernel_sel,       // Select one of 4 kernels
    output logic signed [7:0] kernel_out [0:8]  // 3x3 kernel weights
);

    // Define 4 kernels, each with 9 weights
    logic signed [7:0] kernels [0:3][0:8];

    // Initialize all 4 kernels
    initial begin
        // Kernel 0 - Sobel X
        kernels[0] = '{-1,  0,  1,
                      -2,  0,  2,
                      -1,  0,  1};

        // Kernel 1 - Sobel Y
        kernels[1] = '{-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

        // Kernel 2 - Sharpen
        kernels[2] = '{ 0, -1,  0,
                      -1,  5, -1,
                       0, -1,  0};

        // Kernel 3 - Box Blur
        kernels[3] = '{1, 1, 1,
                      1, 1, 1,
                      1, 1, 1};  // Divide output later by 9 or shift
    end

    // On every clock, output the selected kernel
    always_ff @(posedge clk) begin
        kernel_out <= kernels[kernel_sel];
    end

endmodule
