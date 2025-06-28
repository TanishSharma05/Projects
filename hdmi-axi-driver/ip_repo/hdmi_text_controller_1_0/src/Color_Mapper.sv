module  color_mapper ( input  logic [9:0] BallX, BallY, DrawX, DrawY, Ball_size,
                       output logic [3:0]  Red, Green, Blue,
                       input logic [15:0] char,
                       input logic [31:0] palette [8],
                       input logic is_inverted,
                       input logic [31:0] ctrl_reg);
    
    logic curr_pix;
    logic [10:0] font_rom_address;
    logic [7:0] font_data;
    
    
    logic [3:0] background_red;
    logic [3:0] background_green;
    logic [3:0] background_blue;
    
    logic [3:0] foreground_red;
    logic [3:0] foreground_green;
    logic [3:0] foreground_blue;
    
    logic [2:0] pix_col;
    
    logic inv_check;
    
    always_comb
    begin:RGB_Display
        
//        foreground_red = ctrl_reg[24:21];
//        foreground_green = ctrl_reg[20:17];
//        foreground_blue = ctrl_reg[16:13];
        
//        background_red = ctrl_reg[12:9];
//        background_green = ctrl_reg[8:5];
//        background_blue = ctrl_reg[4:1];

        case(char[7:4])
        
            4'b0000: begin
                foreground_red <= palette[0][12:9];
                foreground_green <= palette[0][8:5];
                foreground_blue <= palette[0][4:1];
                
                
            end
            
            4'b0001: begin
                foreground_red <= palette[0][24:21];
                foreground_green <= palette[0][20:17];
                foreground_blue <= palette[0][16:13];
            
            end
            
            4'b0010: begin
                foreground_red <= palette[1][12:9];
                foreground_green <= palette[1][8:5];
                foreground_blue <= palette[1][4:1];
            
            end
            
            4'b0011: begin
                foreground_red <= palette[1][24:21];
                foreground_green <= palette[1][20:17];
                foreground_blue <= palette[1][16:13];
            end
            
            4'b0100: begin
                foreground_red <= palette[2][12:9];
                foreground_green <= palette[2][8:5];
                foreground_blue <= palette[2][4:1];
            
            end
            
            4'b0101: begin
                foreground_red <= palette[2][24:21];
                foreground_green <= palette[2][20:17];
                foreground_blue <= palette[2][16:13];
            
            end
            
            4'b0110: begin
                foreground_red <= palette[3][12:9];
                foreground_green <= palette[3][8:5];
                foreground_blue <= palette[3][4:1];
            
            end
            
            4'b0111: begin
                foreground_red <= palette[3][24:21];
                foreground_green <= palette[3][20:17];
                foreground_blue <= palette[3][16:13];
            
            end
            
            4'b1000: begin
                foreground_red <= palette[4][12:9];
                foreground_green <= palette[4][8:5];
                foreground_blue <= palette[4][4:1];
            
            end
            
            4'b1001: begin
                foreground_red <= palette[4][24:21];
                foreground_green <= palette[4][20:17];
                foreground_blue <= palette[4][16:13];
            
            end
            
            4'b1010: begin
                foreground_red <= palette[5][12:9];
                foreground_green <= palette[5][8:5];
                foreground_blue <= palette[5][4:1];
            
            end
            
            4'b1011: begin
                foreground_red <= palette[5][24:21];
                foreground_green <= palette[5][20:17];
                foreground_blue <= palette[5][16:13];
            
            end
            
            4'b1100: begin
                foreground_red <= palette[6][12:9];
                foreground_green <= palette[6][8:5];
                foreground_blue <= palette[6][4:1];
            end
            
            4'b1101: begin
                foreground_red <= palette[6][24:21];
                foreground_green <= palette[6][20:17];
                foreground_blue <= palette[6][16:13];
            end
            
            4'b1110: begin
                foreground_red <= palette[7][12:9];
                foreground_green <= palette[7][8:5];
                foreground_blue <= palette[7][4:1];
            end
            
            4'b1111: begin
                foreground_red <= palette[7][24:21];
                foreground_green <= palette[7][20:17];
                foreground_blue <= palette[7][16:13];
            end
            
            default : ;
            
		endcase
		
		 case(char[3:0])
        
            4'b0000: begin
                background_red <= palette[0][12:9];
                background_green <= palette[0][8:5];
                background_blue <= palette[0][4:1];
                   
            end
            
            4'b0001: begin
                background_red <= palette[0][24:21];
                background_green <= palette[0][20:17];
                background_blue <= palette[0][16:13];
            
            end
            
            4'b0010: begin
                background_red <= palette[1][12:9];
                background_green <= palette[1][8:5];
                background_blue <= palette[1][4:1];
            
            end
            
            4'b0011: begin
                background_red <= palette[1][24:21];
                background_green <= palette[1][20:17];
                background_blue <= palette[1][16:13];
            end
            
            4'b0100: begin
                background_red <= palette[2][12:9];
                background_green <= palette[2][8:5];
                background_blue <= palette[2][4:1];
            
            end
            
            4'b0101: begin
                background_red <= palette[2][24:21];
                background_green <= palette[2][20:17];
                background_blue <= palette[2][16:13];
            
            end
            
            4'b0110: begin
                background_red <= palette[3][12:9];
                background_green <= palette[3][8:5];
                background_blue <= palette[3][4:1];
            
            end
            
            4'b0111: begin
                background_red <= palette[3][24:21];
                background_green <= palette[3][20:17];
                background_blue <= palette[3][16:13];
            
            end
            
            4'b1000: begin
                background_red <= palette[4][12:9];
                background_green <= palette[4][8:5];
                background_blue <= palette[4][4:1];
            
            end
            
            4'b1001: begin
                background_red <= palette[4][24:21];
                background_green <= palette[4][20:17];
                background_blue <= palette[4][16:13];
            
            end
            
            4'b1010: begin
                background_red <= palette[5][12:9];
                background_green <= palette[5][8:5];
                background_blue <= palette[5][4:1];
            
            end
            
            4'b1011: begin
                background_red <= palette[5][24:21];
                background_green <= palette[5][20:17];
                background_blue <= palette[5][16:13];
            
            end
            
            4'b1100: begin
                background_red <= palette[6][12:9];
                background_green <= palette[6][8:5];
                background_blue <= palette[6][4:1];
            end
            
            4'b1101: begin
                background_red <= palette[6][24:21];
                background_green <= palette[6][20:17];
                background_blue <= palette[6][16:13];
            end
            
            4'b1110: begin
                background_red <= palette[7][12:9];
                background_green <= palette[7][8:5];
                background_blue <= palette[7][4:1];
            end
            
            4'b1111: begin
                background_red <= palette[7][24:21];
                background_green <= palette[7][20:17];
                background_blue <= palette[7][16:13];
            end
            
            default : ;
            
		endcase
        
    end
    
        
        
        
    always_comb begin
          
        font_rom_address = (char[14:8] << 4) + DrawY[3:0];
        
        pix_col = DrawX[2:0];
        
        curr_pix = font_data[7 - pix_col];
    end
    
    font_rom font_rom_inst (

            .addr(font_rom_address),
            .data(font_data)
        );
        

    always_comb begin
        
        
       
        
        inv_check = curr_pix ^ is_inverted;
        
        case(inv_check)
        
            1'b0: begin
                Red = background_red;
                Green = background_green;
                Blue = background_blue;
            end
            
            1'b1: begin
                Red = foreground_red;
                Green = foreground_green;
                Blue = foreground_blue;
            end
         endcase
         
     end
            
        
        
        
        
        
    
    
    
	 
 /* Old Ball: Generated square box by checking if the current pixel is within a square of length
    2*BallS, centered at (BallX, BallY).  Note that this requires unsigned comparisons.
	 
    if ((DrawX >= BallX - Ball_size) &&
       (DrawX <= BallX + Ball_size) &&
       (DrawY >= BallY - Ball_size) &&
       (DrawY <= BallY + Ball_size))
       )

     New Ball: Generates (pixelated) circle by using the standard circle formula.  Note that while 
     this single line is quite powerful descriptively, it causes the synthesis tool to use up three
     of the 120 available multipliers on the chip!  Since the multiplicants are required to be signed,
	  we have to first cast them from logic to int (signed by default) before they are multiplied). */
	  
//    int DistX, DistY, Size;
//    assign DistX = DrawX - BallX;
//    assign DistY = DrawY - BallY;
//    assign Size = Ball_size;
  
//    always_comb 
//    begin:Ball_on_proc
//        if ( (DistX*DistX + DistY*DistY) <= (Size * Size) )
//            ball_on = 1'b1;
//        else 
//            ball_on = 1'b0;
//     end 
       
//    always_comb
//    begin:RGB_Display
//        if ((ball_on == 1'b1)) begin 
//            Red = 4'hf;
//            Green = 4'h7;
//            Blue = 4'h0;
//        end       
//        else begin 
//            Red = 4'hf - DrawX[9:6]; 
//            Green = 4'hf - DrawX[9:6];
//            Blue = 4'hf - DrawX[9:6];
//        end      

endmodule
