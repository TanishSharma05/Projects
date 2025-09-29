`timescale 1ns / 1ps

module top_module_tb;

    // ----------------------------------------------------------------
    // DUT I/O
    // ----------------------------------------------------------------
    logic        clk;
    logic        reset;
    logic        start;
    logic        tb_write_en;
    logic [9:0]  tb_write_addr;  // 10-bit for 0..783
    logic [7:0]  tb_write_data;
    logic [7:0]  led_out;

    // 100 MHz clock (10 ns period)
    always #5 clk = ~clk;

    // ----------------------------------------------------------------
    // Instantiate DUT
    // ----------------------------------------------------------------
    top_module #(
        .IMG_W(28),
        .IMG_H(28)
    ) uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .led_out(led_out),
        .tb_write_en(tb_write_en),
        .tb_write_addr(tb_write_addr),
        .tb_write_data(tb_write_data)
    );

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------
    task automatic write_pixel(input int addr, input byte val);
        begin
            tb_write_en   = 1'b1;
            tb_write_addr = addr[9:0];
            tb_write_data = val;
            @(negedge clk);
        end
    endtask

    // Create a simple 28x28 test image:
    // pixel(x,y) = clamp((x*8 + y*4), 0..255)
    function automatic byte gen_px(input int x, input int y);
        int v;
        begin
            v = (x*8 + y*4);
            if (v > 255) v = 255;
            gen_px = byte'(v[7:0]);
        end
    endfunction

    // ----------------------------------------------------------------
    // Wave dump (for GTKWave / Questa)
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("top_module_tb.vcd");
        $dumpvars(0, top_module_tb);
    end

    // ----------------------------------------------------------------
    // Main stimulus
    // ----------------------------------------------------------------
    initial begin
        $display("=== 28x28 CNN Testbench Starting ===");

        // Init
        clk          = 1'b0;
        reset        = 1'b1;
        start        = 1'b0;
        tb_write_en  = 1'b0;
        tb_write_addr= '0;
        tb_write_data= '0;

        // Hold reset for a bit
        repeat (4) @(negedge clk);
        reset = 1'b0;
        repeat (4) @(negedge clk);

        // ----------------------------------------------------------------
        // Load 28x28 image into BRAM addresses 0..783 (row-major)
        // ----------------------------------------------------------------
        $display("[%0t] Loading 28x28 image into BRAM...", $time);
        for (
