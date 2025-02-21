`define INIT 3'b00
`define PLAY 3'b01
`define WAIT_RESULT 3'b10


module top(
    input clk, 
    input reset, 
    inout PS2_DATA, 
    inout PS2_CLK,
    input coin_push,
    output left_motor,
    output right_motor,
    output reg [1:0]left,
    output reg [1:0]right,
    output front_motor,
    output reg [1:0] front,
    output reg[3:0]LED,
    output reg [3:0]AN,
    output reg [6:0]display
    );
    
    parameter [8:0] UP = 9'b0_0111_0101, DOWN =  9'b0_0111_0010, ENTER = 9'b0_0101_1010, SPACE = 9'b0_0010_1001;


    wire db_reset, one_reset;
    wire db_push, one_push;
    wire [511:0] key_down;
	wire [8:0] last_change; //push which keyboard
	wire been_ready;
	wire pulse_been_ready, db_ready;
    reg kup, kdown;
    
    Debounce de0(clk, reset, db_reset);
    OnePulse op0(clk, db_reset, one_reset);
    Debounce de1(clk, coin_push, db_push);
    OnePulse op1(clk, db_push, one_push);
    
    motor A(
        .clk(clk),
        .rst(one_reset),
        .pwm(left_motor)
    );
    
    motor B(
        .clk(clk),
        .rst(one_reset),
        .pwm(front_motor)
    );
    
    KeyboardDecoder key_de (
		.key_down(key_down),
		.last_change(last_change),
		.key_valid(been_ready),
		.PS2_DATA(PS2_DATA),
		.PS2_CLK(PS2_CLK),
		.rst(one_reset),
		.clk(clk)
	);
    
    assign right_motor = left_motor;
    
    reg [29:0]cnt, next_cnt;
    reg [1:0]state, next_state;
    
    always@(posedge clk) begin
        if (one_reset) begin
            cnt <= 27'b0;
            state <= `INIT;
        end 
        else begin       
            cnt <= next_cnt;
            state <= next_state;
        end
   
    end
    
     reg [3:0] display_num;
    
    always@(*)begin
        case(state)
            `INIT:begin
               //front
               front = 2'b00;
               next_cnt = 27'b0;
               left = 2'b00;
              if(one_push)begin
                    next_state =  `PLAY;
               end
               else begin
                next_state = state;
               end
            end
            `PLAY:begin
            //front
                front = 2'b00;
                if(been_ready && key_down[SPACE] == 1'b1) begin
                    left = 2'b00;
                    next_cnt = 27'b1;
                    next_state = `WAIT_RESULT;     
                    kup = 1'b0;
                    kdown = 1'b0;      
                    LED[3] = 1'b0;   
                    LED[2] = 1'b0;                 
                end
                else if(been_ready && key_down[UP] == 1'b1) begin
                    left = 2'b10;
                    next_cnt = 27'b1;
                    next_state = state;
                    kup = 1'b1;
                    kdown = 1'b0;
                    LED[3] = 1'b1;
                    LED[2] = 1'b0;  
                end
                else if(been_ready && key_down[DOWN] == 1'b1)begin
                    left = 2'b01;
                    next_cnt = 27'b1;
                    next_state = state;
                    kup = 1'b0;
                    kdown = 1'b1;
                    LED[3] = 1'b1;
                    LED[2] = 1'b0;  
                end
                else begin
                    if (kup && !kdown && cnt<=27'd10_000_000) begin
                        left = 2'b10;
                        next_cnt = cnt + 27'b1;
                        kup = kup;
                        kdown = kdown;
                        next_state = state;
                    end
                    else if (!kup && kdown && cnt<=27'd10_000_000) begin
                        left = 2'b01;
                        next_cnt = cnt + 27'b1;
                        kup = kup;
                        kdown = kdown;
                        next_state = state;
                    end
                    else begin
                        left = 2'b00;
                        next_cnt = cnt;
                        kup = 1'b0;
                        kdown = 1'b0;
                    end
                    
                    next_state = state;
                    LED[3] = 1'b0;
                    LED[2] = 1'b1;  
                end
            end
            `WAIT_RESULT: begin 
            //front, state n cnt
                if (cnt<=30'd7500_0000) begin
                    front = 2'b01;
                    next_state = `WAIT_RESULT;
                    next_cnt = cnt+27'b1;
                end
                else if (cnt>30'd7500_0000 && cnt<=30'd20000_0000)begin
                    front = 2'b00;
                    next_state = `WAIT_RESULT;
                    next_cnt = cnt+27'b1;
                end
                else if (cnt > 30'd20000_0000 && cnt<=30'd27500_0000)begin
                    front = 2'b10;
                    next_state = `WAIT_RESULT;
                    next_cnt = cnt+27'b1;
                end         
                else begin
                    front = 2'b11;
                    next_state = `INIT;
                    next_cnt = 27'b0;
                end
             //up
             left = 2'b00;
             end
             
             
             default:begin
                front = 2'b00;
                next_state = `INIT;
                next_cnt = 27'b0;
                left = 2'b00;
             end
        endcase  
        right = left;
          
    end
    
    always@(posedge clk)begin
    if(state == `INIT) begin
            display_num <= 1'b0;
            AN <= 4'b1110;
     end 
     else if(state==`PLAY)begin
            AN <= 4'b1110;  
             if(left == 2'b10) begin
                display_num <= 4'd7;
            end
            else if(left == 2'b01)begin
                display_num <= 4'd8;
            end
            else begin
                display_num <= 4'd9;
            end
     end
     else if(state == `WAIT_RESULT) begin 
            AN <= 4'b1110;
            display_num <= 4'd2;
     end
     else begin 
            AN <= 4'b1110;
            display_num <= 4'd10;
     end
    end
    
    always@(*) begin
        case(display_num)
            4'd0: display = 7'b1000000; 
            4'd1: display = 7'b1111001;
            4'd2: display = 7'b0100100;
            4'd3: display = 7'b0110000;
            4'd4: display = 7'b0011001;
            4'd5: display = 7'b0010010;
            4'd6: display = 7'b0000010;
            4'd7: display = 7'b1111000;
            4'd8: display = 7'b0000000;
            4'd9: display = 7'b0010000;
            default: display = 7'b1111111;
        endcase
    end
    
    //assign LED[1] = front[1];
    //assign LED[0] = front[0];
endmodule



module motor(
    input clk,
   input rst,
    output pwm
);

    reg [9:0]next_left_motor;//, next_right_motor;
    reg [9:0]left_motor;//, right_motor;
    wire left_pwm;//, right_pwm;

    motor_pwm m0(clk, left_motor, left_pwm);
    
    always@(posedge clk)begin
       if(rst)begin
            left_motor <= next_left_motor;
        end
    end
    
    // [TO-DO] take the right speed for different situation
    always @(*) begin
        next_left_motor = 10'd50;
    end
    assign pwm = left_pwm;

endmodule

module motor_pwm (
    input clk,
    input reset,
    input [9:0]duty,
	output pmod_1 //PWM
);
        
    PWM_gen pwm_0 ( 
        .clk(clk), 
        .reset(reset), 
        .freq(32'd25000),
        .duty(duty), 
        .PWM(pmod_1)
    );

endmodule

//generte PWM by input frequency & duty
module PWM_gen (
    input wire clk,
    input wire reset,
	input [31:0] freq,
    input [9:0] duty,
    output reg PWM
);
    wire [31:0] count_max = 32'd100_000_000 / freq;
    wire [31:0] count_duty = count_max * duty / 32'd1024;
    reg [31:0] count;
        
    always @(posedge clk) begin
        if (reset) begin
            count <= 32'b0;
            PWM <= 1'b0;
        end 
        else if (count < count_max) begin
            count <= count + 32'd1;
            if(count < count_duty)
                PWM <= 1'b1;
            else
                PWM <= 1'b0;
        end else begin
            count <= 32'b0;
            PWM <= 1'b0;
        end
    end
endmodule
module KeyboardDecoder(
	output reg [511:0] key_down,
	output wire [8:0] last_change,
	output reg key_valid,
	inout wire PS2_DATA,
	inout wire PS2_CLK,
	input wire rst,
	input wire clk
    );
    parameter [1:0] INIT			= 2'b00;
    parameter [1:0] WAIT_FOR_SIGNAL = 2'b01;
    parameter [1:0] GET_SIGNAL_DOWN = 2'b10;
    parameter [1:0] WAIT_RELEASE    = 2'b11;
	parameter [7:0] IS_INIT			= 8'hAA;
    parameter [7:0] IS_EXTEND		= 8'hE0;
    parameter [7:0] IS_BREAK		= 8'hF0;
    reg [9:0] key;		// key = {been_extend, been_break, key_in}
    reg [1:0] state;
    reg been_ready, been_extend, been_break;
    
    wire [7:0] key_in;
    wire is_extend;
    wire is_break;
    wire valid;
    wire err;
    wire pulse_been_ready;
    
    wire [511:0] key_decode = 1 << last_change;
    assign last_change = {key[9], key[7:0]};
    
    KeyboardCtrl_0 inst (
		.key_in(key_in),
		.is_extend(is_extend),
		.is_break(is_break),
		.valid(valid),
		.err(err),
		.PS2_DATA(PS2_DATA),
		.PS2_CLK(PS2_CLK),
		.rst(rst),
		.clk(clk)
	);
	
	OnePulse op (
		.pb_one_pulse(pulse_been_ready),
		.pb_debounce(been_ready),
		.clk(clk)
	);
    
    always @ (posedge clk, posedge rst) begin
    	if (rst) begin
    		state <= INIT;
    		been_ready  <= 1'b0;
    		been_extend <= 1'b0;
    		been_break  <= 1'b0;
    		key <= 10'b0_0_0000_0000;
    	end else begin
    		state <= state;
			been_ready  <= been_ready;
			been_extend <= (is_extend) ? 1'b1 : been_extend;
			been_break  <= (is_break ) ? 1'b1 : been_break;
			key <= key;
    		case (state)
    			INIT : begin
    					if (key_in == IS_INIT) begin
    						state <= WAIT_FOR_SIGNAL;
    						been_ready  <= 1'b0;
							been_extend <= 1'b0;
							been_break  <= 1'b0;
							key <= 10'b0_0_0000_0000;
    					end else begin
    						state <= INIT;
    					end
    				end
    			WAIT_FOR_SIGNAL : begin
    					if (valid == 0) begin
    						state <= WAIT_FOR_SIGNAL;
    						been_ready <= 1'b0;
    					end else begin
    						state <= GET_SIGNAL_DOWN;
    					end
    				end
    			GET_SIGNAL_DOWN : begin
						state <= WAIT_RELEASE;
						key <= {been_extend, been_break, key_in};
						been_ready  <= 1'b1;
    				end
    			WAIT_RELEASE : begin
    					if (valid == 1) begin
    						state <= WAIT_RELEASE;
    					end else begin
    						state <= WAIT_FOR_SIGNAL;
    						been_extend <= 1'b0;
    						been_break  <= 1'b0;
    					end
    				end
    			default : begin
    					state <= INIT;
						been_ready  <= 1'b0;
						been_extend <= 1'b0;
						been_break  <= 1'b0;
						key <= 10'b0_0_0000_0000;
    				end
    		endcase
    	end
    end
    
    always @ (posedge clk, posedge rst) begin
    	if (rst) begin
    		key_valid <= 1'b0;
    		key_down <= 511'b0;
    	end else if (key_decode[last_change] && pulse_been_ready) begin
    		key_valid <= 1'b1;
    		if (key[8] == 0) begin
    			key_down <= key_down | key_decode;
    		end else begin
    			key_down <= key_down & (~key_decode);
    		end
    	end else begin
    		key_valid <= 1'b0;
			key_down <= key_down;
    	end
    end

endmodule



module Debounce(clk, pb, pb_debounce);
    input clk, pb;
    output pb_debounce;
    reg [5:0] DFF;
    always @(posedge clk) begin
        DFF[5:1] <= DFF[4:0];
        DFF[0] <= pb;
    end
    assign pb_debounce = (DFF == 6'b111111)? 1'b1: 1'b0;
endmodule

module OnePulse(clk, pb_debounce, pb_one_pulse);
    input clk, pb_debounce;
    output reg pb_one_pulse;
    reg pb_debounce_delay;
    always@(posedge clk) begin
        pb_one_pulse <= pb_debounce & (!pb_debounce_delay);
        pb_debounce_delay <= pb_debounce;
    end
endmodule

