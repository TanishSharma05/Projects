module pooling_unit #(
    parameter MODE = "MAX"  // "MAX" or "AVG"
) (
    input  logic clk,
    input  logic signed [15:0] in0,  // Top-left
    input  logic signed [15:0] in1,  // Top-right
    input  logic signed [15:0] in2,  // Bottom-left
    input  logic signed [15:0] in3,  // Bottom-right
    input  logic valid_in,

    output logic signed [15:0] pooled_out,
    output logic valid_out
);

    logic signed [15:0] max_1, max_2;
    logic signed [15:0] max_final;
    logic signed [15:0] avg;

    // Combinational pooling logic
    always_comb begin
        // Max Pooling
        max_1 = (in0 > in1) ? in0 : in1;
        max_2 = (in2 > in3) ? in2 : in3;
        max_final = (max_1 > max_2) ? max_1 : max_2;

        // Average Pooling: (in0 + in1 + in2 + in3) / 4
        avg = (in0 + in1 + in2 + in3) >>> 2;  // Arithmetic shift right by 2 (divide by 4)
    end

    // Register output
    always_ff @(posedge clk) begin
        if (valid_in) begin
            pooled_out <= (MODE == "MAX") ? max_final : avg;
            valid_out  <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
