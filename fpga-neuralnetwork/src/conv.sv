module conv (
    input  logic clk,
    input  logic signed [7:0] pixel_in [0:8],
    input  logic signed [7:0] kernel_in [0:8],
    input  logic              load,

    output logic signed [15:0] conv_out,
    output logic               valid
);

    logic signed [7:0] reg_pixel  [0:8];
    logic signed [7:0] reg_kernel [0:8];
    logic              calc;

    // Stage 1: Latch on load
    always_ff @(posedge clk) begin
        if (load) begin
            for (int i = 0; i < 9; i++) begin
                reg_pixel[i]  <= pixel_in[i];
                reg_kernel[i] <= kernel_in[i];
            end
            calc <= 1;
        end else begin
            calc <= 0;
        end
    end

    // Stage 2: Compute on next clock
    logic signed [15:0] mult_result [0:8];
    logic signed [15:0] sum;

    always_comb begin
        for (int i = 0; i < 9; i++)
            mult_result[i] = reg_pixel[i] * reg_kernel[i];

        sum = mult_result[0] + mult_result[1] + mult_result[2] +
              mult_result[3] + mult_result[4] + mult_result[5] +
              mult_result[6] + mult_result[7] + mult_result[8];
    end

    always_ff @(posedge clk) begin
        if (calc) begin
            conv_out <= sum;
            valid    <= 1;
        end else begin
            valid    <= 0;
        end
    end

endmodule
