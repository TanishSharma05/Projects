module top_module #(
    // ---- Global geometry ----
    parameter int IMG_W      = 28,
    parameter int IMG_H      = 28,
    parameter int K          = 3,                 // 3x3 kernel
    parameter int CONV_W     = IMG_W - K + 1,     // 26
    parameter int CONV_H     = IMG_H - K + 1,     // 26
    parameter int POOL_K     = 2,
    parameter int POOL_S     = 2,
    parameter int POOL_W     = CONV_W / POOL_S,   // 13
    parameter int POOL_H     = CONV_H / POOL_S,   // 13
    parameter int ACT_PIXELS = CONV_W * CONV_H,   // 676
    parameter int POOL_PIX   = POOL_W * POOL_H    // 169
) (
    input  logic        clk,
    input  logic        reset,
    input  logic        start,
    output logic [7:0]  led_out,

    // For simulation image loading
    input  logic        tb_write_en,
    input  logic [9:0]  tb_write_addr, // 2^10=1024 > 784
    input  logic [7:0]  tb_write_data
);

    // Control signals
    logic [$clog2(CONV_H)-1:0] conv_row;
    logic [$clog2(CONV_W)-1:0] conv_col;
    logic        patch_start_pulse;
    logic        begin_pooling;
    logic        begin_flatten;
    logic        begin_fc;

    // Data wires
    logic signed [7:0] kernel_data [0:K*K-1];
    logic signed [7:0] patch_values [0:K*K-1];

    logic signed [15:0] conv_result;
    logic               conv_valid;

    logic signed [15:0] activated;
    logic               relu_valid;

    // Activation map buffer (CONV_W*CONV_H)
    logic signed [15:0] act_map [0:ACT_PIXELS-1];
    logic [$clog2(ACT_PIXELS):0] act_wr_idx;
    logic act_map_full;

    // Pool window I/O
    logic signed [15:0] pool_in0, pool_in1, pool_in2, pool_in3;
    logic               pool_win_valid;
    logic               pool_sweep_done;

    logic signed [15:0] pooled_out;
    logic               pooled_valid;

    // Pooled map buffer (13x13 = 169)
    logic signed [15:0] pooled_map [0:POOL_PIX-1];
    logic [$clog2(POOL_PIX):0] pooled_wr_idx;
    logic pooled_map_full;

    // Flatten/FC
    logic signed [15:0] flat_val;
    logic               flat_valid;
    logic               flatten_done;

    logic signed [15:0] fc_out [0:1];
    logic               fc_valid;

    logic [0:0] class_result;
    logic       result_valid;

    // --- Image memory (28x28) ---
    logic [9:0]  img_read_addr;
    logic [7:0]  img_read_data;

    bram_image_memory #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(10)   // 0..1023 (784 used)
    ) img_mem (
        .clk(clk),
        .write_en(tb_write_en),
        .write_addr(tb_write_addr),
        .write_data(tb_write_data),
        .read_addr(img_read_addr),
        .read_data(img_read_data)
    );

    // --- Controller: iterates conv positions & phases ---
    controller_fsm #(
        .IMG_W(IMG_W), .IMG_H(IMG_H), .K(K)
    ) ctrl (
        .clk(clk),
        .reset(reset),
        .start(start),
        .patch_valid(relu_valid),     // advance after ReLU captured
        .conv_row(conv_row),
        .conv_col(conv_col),
        .patch_start(patch_start_pulse),
        .begin_pool(begin_pooling),
        .begin_flatten(begin_flatten),
        .begin_fc(begin_fc)
    );

    // --- Patch extractor @ 28 stride ---
    patch_extractor #(
        .IMG_W(IMG_W), .K(K)
    ) patch_gen (
        .clk(clk),
        .reset(reset),
        .start(patch_start_pulse),
        .base_addr(conv_row*IMG_W + conv_col),
        .memory_data(img_read_data),
        .memory_addr(img_read_addr),
        .valid_out(/* not used; conv uses patch_start_pulse */),
        .patch(patch_values)
    );

    // --- Kernels ---
    logic [1:0] kernel_sel;
    assign kernel_sel = 2'd0; // keep Sobel X by default; can drive from TB or FSM
    kernel_bank kbank (
        .clk(clk),
        .kernel_sel(kernel_sel),
        .kernel_out(kernel_data)
    );

    // --- Conv ---
    conv conv_core (
        .clk(clk),
        .pixel_in(patch_values),
        .kernel_in(kernel_data),
        .load(patch_start_pulse),   // one patch per pulse
        .conv_out(conv_result),
        .valid(conv_valid)
    );

    // --- ReLU ---
    activation_unit relu (
        .clk(clk),
        .in_data(conv_result),
        .valid_in(conv_valid),
        .out_data(activated),
        .valid_out(relu_valid)
    );

    // --- Write ReLU into activation map [row,col] ---
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            act_wr_idx   <= '0;
        end else if (relu_valid) begin
            act_map[conv_row*CONV_W + conv_col] <= activated;
            if (conv_row==CONV_H-1 && conv_col==CONV_W-1)
                act_wr_idx <= ACT_PIXELS; // just a marker
        end
    end
    assign act_map_full = (act_wr_idx == ACT_PIXELS);

    // --- Pooling window sweep over 26x26 map -> 13x13 outputs ---
    pooling_window_controller #(
        .ACT_W(CONV_W), .ACT_H(CONV_H), .POOL_K(POOL_K), .POOL_S(POOL_S)
    ) window_ctrl (
        .clk(clk),
        .reset(reset),
        .act_map(act_map),
        .start(begin_pooling),
        .out0(pool_in0), .out1(pool_in1), .out2(pool_in2), .out3(pool_in3),
        .valid_out(pool_win_valid),
        .done(pool_sweep_done)
    );

    pooling_unit #(.MODE("MAX")) pool (
        .clk(clk),
        .in0(pool_in0), .in1(pool_in1), .in2(pool_in2), .in3(pool_in3),
        .valid_in(pool_win_valid),
        .pooled_out(pooled_out),
        .valid_out(pooled_valid)
    );

    // --- Capture 13x13 pooled map ---
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            pooled_wr_idx <= '0;
        end else if (pooled_valid) begin
            pooled_map[pooled_wr_idx] <= pooled_out;
            pooled_wr_idx <= pooled_wr_idx + 1;
        end
    end
    assign pooled_map_full = (pooled_wr_idx == POOL_PIX);

    // --- Flatten 169 → stream ---
    flattener #(.N(POOL_PIX)) flat (
        .clk(clk),
        .reset(reset),
        .start(begin_flatten & pooled_map_full),
        .pooled_map(pooled_map),
        .fc_input(flat_val),
        .valid_out(flat_valid),
        .done(flatten_done)
    );

    // --- FC: consumes the whole 169-vector at once (buffered), then valid ---
    // Buffer the stream into a vector expected by FC
    logic signed [15:0] fc_input_vec [0:POOL_PIX-1];
    logic [$clog2(POOL_PIX):0] fc_buf_idx;
    logic fc_inputs_ready;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            fc_buf_idx     <= '0;
            fc_inputs_ready<= 1'b0;
        end else begin
            if (flat_valid) begin
                fc_input_vec[fc_buf_idx] <= flat_val;
                if (fc_buf_idx == POOL_PIX-1) begin
                    fc_inputs_ready <= 1'b1;
                    fc_buf_idx      <= '0;
                end else begin
                    fc_buf_idx      <= fc_buf_idx + 1;
                end
            end
            if (fc_valid) fc_inputs_ready <= 1'b0; // consumed
        end
    end

    fc_layer #(.N_IN(POOL_PIX), .N_OUT(2)) fc (
        .clk(clk),
        .reset(reset),
        .input_vec(fc_input_vec),
        .valid_in(begin_fc & fc_inputs_ready),
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
