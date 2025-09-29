`timescale 1ns / 1ps

module top_module_tb;

    // DUT inputs
    logic clk;
    logic reset;
    logic start;
    logic tb_write_en;
    logic [3:0] tb_write_addr;
    logic [7:0] tb_write_data;

    // DUT outputs
    logic [7:0] led_out;

    // Clock generation
    always #5 clk = ~clk;

    // Instantiate DUT
    top_module uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .led_out(led_out),
        .tb_write_en(tb_write_en),
        .tb_write_addr(tb_write_addr),
        .tb_write_data(tb_write_data)
    );

    // Sample 4x4 image
    logic [7:0] test_image [0:15] = '{
        8'h0C, 8'h22, 8'h38, 8'h4E,
        8'h5A, 8'h2D, 8'h43, 8'h59,
        8'h17, 8'h2C, 8'h42, 8'h58,
        8'h0B, 8'h16, 8'h21, 8'h2C
    };

    initial begin
        $display("=== CNN Sliding Testbench Starting ===");

        // Initialize
        clk = 0;
        reset = 1;
        start = 0;
        tb_write_en = 0;

        // Hold reset
        #20;
        reset = 0;
        #20;

        // Load image into BRAM
        for (int i = 0; i < 16; i++) begin
            tb_write_en   = 1;
            tb_write_addr = i[3:0];
            tb_write_data = test_image[i];
            #10;
        end

        tb_write_en = 0;
        #20;

        // Trigger start
        start = 1;
        #10;
        start = 0;

        // Simulation run
        repeat (3000) begin
            @(posedge clk);

            // Debug info
            if (uut.patch_valid)
                $display("[Patch] @ %0t ns: Top-left Addr = %0d", $time, uut.pixel_addr);

            if (uut.conv_valid)
                $display("[Convolution] Result = %0d", uut.conv_result);

            if (uut.relu_valid)
                $display("[ReLU] Activated = %0d", uut.activated);

            if (uut.pool_valid_out)
                $display("[Pooling] Out = %0d", uut.pooled_out);

            if (uut.flat_valid_out)
                $display("[Flatten] Out = %0d", uut.fc_patch);

            if (uut.fc_valid)
                $display("[FC Layer] Out0 = %0d, Out1 = %0d", uut.fc_out[0], uut.fc_out[1]);

            if (uut.result_valid)
                $display("[Argmax] Class = %0d => LED = %b", uut.class_result, led_out);
        end

        $display("=== Simulation Complete ===");
        $stop;
    end

endmodule 