module led_driver (
    input  logic clk,
    input  logic reset,
    input  logic [0:0] class_idx,   // Final predicted class (0 or 1)
    input  logic       valid_in,    // One-cycle pulse from argmax

    output logic [7:0] led_out      // Output to 8 board LEDs
);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            led_out <= 8'b00000000;
        end else if (valid_in) begin
            // Light up one LED based on the class index
            case (class_idx)
                1'b0: led_out <= 8'b00000001;  // LED0 on ? class 0
                1'b1: led_out <= 8'b00000010;  // LED1 on ? class 1
                default: led_out <= 8'b00000000;
            endcase
        end
    end

endmodule
