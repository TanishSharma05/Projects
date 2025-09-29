module top_module (
    input  logic clk,
    input  logic reset,
    input  logic start,
    output logic [7:0] led_out,

    // For simulation image loading
    input  logic        tb_write_en,
    input  logic [3:0]  tb_write_addr,
    input  logic [7:0]  tb_write_data
);

    // Control signals
    logic [3:0] pixel_addr;
    logic [1:0] kernel_sel;
    logic       load_conv, load_pool, load_fc;

    // Data wires
    logic signed [7:0] kernel_data [0:8];
    logic signed [7:0] patch_values [0:8];

    logic signed [15:0] conv_result;
    logic               conv_valid;

    logic signed [15:0] activated;
    logic               relu_valid;

    logic signed [15:0] pool_in0, pool_in1, pool_in2, pool_in3;
    logic signed [15:0] pooled_out;
    logic               pool_valid_out;
    logic               pool_to_flat_valid;

    logic signed [15:0] pooled_buffer [0:3];
    logic signed [15:0] fc_patch;
    logic signed [15:0] fc_input_buffer [0:3];
    logic               flat_valid_out;
    logic               flatten_done;

    logic [1:0] pool_write_index;
    logic       pooled_map_ready;
    logic [1:0] fc_write_index;
    logic       fc_input_ready;

    logic signed [15:0] fc_out [0:1];
    logic               fc_valid;

    logic [0:0] class_result;
    logic       result_valid;

    logic [3:0] patch_read_addr;
    logic [3:0] image_write_addr;
    logic [7:0] image_write_data;
    logic       image_write_en;
    logic [7:0] image_pixel;

    assign image_write_en   = tb_write_en;
    assign image_write_addr = tb_write_addr;
    assign image_write_data = tb_write_data;

    // === Module Instantiations ===

    controller_fsm ctrl (
    .clk(clk),
    .reset(reset),
    .start(start),
    .patch_valid(patch_valid), // <- ADD THIS
    .pixel_addr(pixel_addr),
    .kernel_sel(kernel_sel),
    .load_conv(load_conv),
    .load_pool(load_pool),
    .load_fc(load_fc)
    );

    bram_image_memory #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(4)
    ) img_mem (
        .clk(clk),
        .write_en(image_write_en),
        .write_addr(image_write_addr),
        .write_data(image_write_data),
        .read_addr(patch_read_addr),
        .read_data(image_pixel)
    );

    patch_extractor patch_gen (
        .clk(clk),
        .reset(reset),
        .start(load_conv),
        .base_addr(pixel_addr),
        .memory_data(image_pixel),
        .memory_addr(patch_read_addr),
        .valid_out(patch_valid),
        .patch(patch_values)
    );

    kernel_bank kbank (
        .clk(clk),
        .kernel_sel(kernel_sel),
        .kernel_out(kernel_data)
    );

    conv conv_core (
        .clk(clk),
        .pixel_in(patch_values),
        .kernel_in(kernel_data),
        .load(patch_valid),
        .conv_out(conv_result),
        .valid(conv_valid)
    );

    activation_unit relu (
        .clk(clk),
        .in_data(conv_result),
        .valid_in(conv_valid),
        .out_data(activated),
        .valid_out(relu_valid)
    );

    pooling_window_controller window_ctrl (
        .clk(clk),
        .reset(reset),
        .start(load_pool),
        .act_map('{activated, activated, activated, activated,
                  activated, activated, activated, activated,
                  activated, activated, activated, activated,
                  activated, activated, activated, activated}),
        .out0(pool_in0),
        .out1(pool_in1),
        .out2(pool_in2),
        .out3(pool_in3),
        .valid_out(pool_valid_out),
        .done()
    );

    pooling_unit #(.MODE("MAX")) pool (
        .clk(clk),
        .in0(pool_in0),
        .in1(pool_in1),
        .in2(pool_in2),
        .in3(pool_in3),
        .valid_in(pool_valid_out),
        .pooled_out(pooled_out),
        .valid_out(pool_to_flat_valid)
    );

    // Pool ? Buffer
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            pool_write_index <= 0;
            pooled_map_ready <= 0;
        end else if (pool_to_flat_valid) begin
            pooled_buffer[pool_write_index] <= pooled_out;
            pool_write_index <= pool_write_index + 1;
            if (pool_write_index == 2'd3)
                pooled_map_ready <= 1;
        end
    end

    flattener flat (
        .clk(clk),
        .reset(reset),
        .start(pooled_map_ready),
        .pooled_map(pooled_buffer),
        .fc_input(fc_patch),
        .valid_out(flat_valid_out),
        .done(flatten_done)
    );

    // Flatten ? FC Buffer
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            fc_write_index <= 0;
            fc_input_ready <= 0;
        end else if (flat_valid_out) begin
            fc_input_buffer[fc_write_index] <= fc_patch;
            fc_write_index <= fc_write_index + 1;
            if (fc_write_index == 2'd3)
                fc_input_ready <= 1;
        end
    end

    fc_layer fc (
        .clk(clk),
        .reset(reset),
        .input_vec(fc_input_buffer),
        .valid_in(fc_input_ready),
        .output_vec(fc_out),
        .valid_out(fc_valid)
    );

    argmax_unit argmax (
        .clk(clk),
        .reset(reset),
        .in0(fc_out[0]),
        .in1(fc_out[1]),
        .valid_in(fc_valid),
        .class_idx(class_result),
        .valid_out(result_valid)
    );

    led_driver leds (
        .clk(clk),
        .reset(reset),
        .class_idx(class_result),
        .valid_in(result_valid),
        .led_out(led_out)
    );

endmodule
