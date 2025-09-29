module flattener (
    input  logic clk,
    input  logic reset,
    input  logic start,                              // Trigger to begin flattening
    input  logic signed [15:0] pooled_map [0:3],     // 2x2 pooled data (4 elements)
    output logic signed [15:0] fc_input,             // 1D output value to FC layer
    output logic valid_out,                          // High when fc_input is valid
    output logic done                                // High when all values sent
);

    logic [1:0] index;      // Index to step through the 4 pooled values
    logic active;           // Flag to indicate flattening is in progress

    // Sequential control logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            index      <= 0;         // Reset index to 0
            active     <= 0;         // Reset active flag
            valid_out  <= 0;         // No output is valid yet
        end else begin
            if (start) begin
                active    <= 1;      // Start flattening
                index     <= 0;      // Start from first element
                valid_out <= 1;      // First output will be valid
            end else if (active) begin
                if (index < 3) begin
                    index <= index + 1;  // Step to next pooled value
                end else begin
                    active    <= 0;      // Done after last value
                    valid_out <= 0;      // No more valid output
                end
            end
        end
    end

    // Output logic: stream out pooled_map[index] each clock
    always_ff @(posedge clk) begin
        if (active)
            fc_input <= pooled_map[index];  // Send one value per clock
    end

    // 'done' is high when we've output the last value
    assign done = (index == 3 && active);

endmodule