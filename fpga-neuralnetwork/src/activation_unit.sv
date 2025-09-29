module activation_unit (
    input  logic clk,
    input  logic signed [15:0] in_data,   // From conv_core
    input  logic               valid_in,  // From conv_core

    output logic signed [15:0] out_data,  // Activated value (ReLU)
    output logic               valid_out  // Output valid signal
);

    always_ff @(posedge clk) begin
        if (valid_in) begin
            out_data  <= (in_data < 0) ? 16'sd0 : in_data;
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
