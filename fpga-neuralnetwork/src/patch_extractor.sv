module patch_extractor (
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic [3:0] base_addr,
    input  logic [7:0] memory_data,
    output logic [3:0] memory_addr,
    output logic       valid_out,
    output logic signed [7:0] patch [0:8]
);

    typedef enum logic [3:0] {
        IDLE,
        READ_0, READ_1, READ_2,
        READ_3, READ_4, READ_5,
        READ_6, READ_7, READ_8,
        DONE,
        HOLD   // Wait for start to go low
    } state_t;

    state_t state, next_state;

    logic signed [7:0] temp_patch [0:8];
    logic [3:0]        cycle_count;
    logic [3:0]        addr_lut [0:8];
    logic              valid_reg;

    // FSM Transition
    always_ff @(posedge clk or posedge reset) begin
        if (reset)
            state <= IDLE;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:     if (start) next_state = READ_0;
            READ_0:   next_state = READ_1;
            READ_1:   next_state = READ_2;
            READ_2:   next_state = READ_3;
            READ_3:   next_state = READ_4;
            READ_4:   next_state = READ_5;
            READ_5:   next_state = READ_6;
            READ_6:   next_state = READ_7;
            READ_7:   next_state = READ_8;
            READ_8:   next_state = DONE;
            DONE:     next_state = HOLD;
            HOLD:     if (!start) next_state = IDLE;
                      else        next_state = HOLD;
            default:  next_state = IDLE;
        endcase
    end

    // Offset LUT
    always_comb begin
        addr_lut[0] = 0;
        addr_lut[1] = 1;
        addr_lut[2] = 2;
        addr_lut[3] = 4;
        addr_lut[4] = 5;
        addr_lut[5] = 6;
        addr_lut[6] = 8;
        addr_lut[7] = 9;
        addr_lut[8] = 10;
    end

    // Address output
    assign memory_addr = base_addr + addr_lut[cycle_count];

    // Patch assembly
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            cycle_count <= 0;
            valid_reg   <= 0;
        end else begin
            valid_reg <= 0;

            if (state >= READ_0 && state <= READ_8) begin
                if (cycle_count > 0)
                    temp_patch[cycle_count - 1] <= memory_data;
                cycle_count <= cycle_count + 1;
            end

            if (state == DONE) begin
                temp_patch[8] <= memory_data;
                valid_reg <= 1;
                cycle_count <= 0;
            end
        end
    end

    assign patch     = temp_patch;
    assign valid_out = valid_reg;

endmodule
