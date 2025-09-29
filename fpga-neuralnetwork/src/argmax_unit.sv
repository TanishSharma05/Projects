module argmax_unit (
    input  logic clk,
    input  logic reset,
    input  logic signed [15:0] in0,     // Class 0 score
    input  logic signed [15:0] in1,     // Class 1 score
    input  logic valid_in,              // Trigger computation

    output logic [0:0] class_idx,       // Output: 0 or 1
    output logic       valid_out        // High when result is ready
);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            class_idx  <= 0;
            valid_out  <= 0;
        end else if (valid_in) begin
            class_idx  <= (in0 > in1) ? 1'b0 : 1'b1;  // Output 0 if in0 > in1, else 1
            valid_out  <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
