module pooling_window_controller (
    input  logic clk,
    input  logic reset,
    input  logic signed [15:0] act_map [0:15],  // 4x4 activation map
    input  logic start,

    output logic signed [15:0] out0, out1, out2, out3,  // 2x2 window values
    output logic              valid_out,
    output logic              done
);

    // FSM states
    typedef enum logic [1:0] {
        IDLE,
        LOAD,
        WAIT,
        NEXT
    } state_t;

    state_t state, next_state;

    // Track top-left pixel of 2x2 window
    logic [3:0] base_index;
    logic [1:0] row, col;

    // FSM sequential
    always_ff @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next_state;
    end

    // FSM combinational
    always_comb begin
        next_state = state;
        valid_out = 0;

        case (state)
            IDLE: if (start) next_state = LOAD;
            LOAD: begin
                valid_out = 1;
                next_state = WAIT;
            end
            WAIT: next_state = NEXT;
            NEXT: if (row == 2 && col == 2) next_state = IDLE;
                  else next_state = LOAD;
        endcase
    end

    // Index management
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            row <= 0;
            col <= 0;
        end else if (state == NEXT) begin
            if (col < 2) col <= col + 1;
            else begin
                col <= 0;
                row <= row + 1;
            end
        end
    end

    // Calculate 1D base index for 2x2 window
    always_comb begin
        base_index = (row * 8) + (col * 2);  // (row * width) + col
    end

    // Output 2x2 window
    always_ff @(posedge clk) begin
        if (state == LOAD) begin
            out0 <= act_map[base_index];               // top-left
            out1 <= act_map[base_index + 1];           // top-right
            out2 <= act_map[base_index + 4];           // bottom-left
            out3 <= act_map[base_index + 5];           // bottom-right
        end
    end

    // Done flag
    assign done = (row == 2 && col == 2 && state == NEXT);

endmodule
