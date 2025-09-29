module image_memory (
    input logic clk,
    input logic [3:0] addr,  // enough for 16 pixels (4x4)
    output logic [7:0] data  // grayscale pixel output
);

    logic [7:0] memory [0:15];  // 4x4 image

    initial begin
        // Hardcoded test image
        memory[0]  = 8'd12;  memory[1]  = 8'd34;  memory[2]  = 8'd56;  memory[3]  = 8'd78;
        memory[4]  = 8'd90;  memory[5]  = 8'd45;  memory[6]  = 8'd67;  memory[7]  = 8'd89;
        memory[8]  = 8'd23;  memory[9]  = 8'd44;  memory[10] = 8'd66;  memory[11] = 8'd88;
        memory[12] = 8'd11;  memory[13] = 8'd22;  memory[14] = 8'd33;  memory[15] = 8'd44;
    end

    always_ff @(posedge clk) begin
        data <= memory[addr];
    end

endmodule