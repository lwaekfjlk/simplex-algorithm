import pdb
import numpy as np
import random
import time
from scipy.optimize import linprog
from simplex import Simplex_Table

class Dual_Simplex_Table(Simplex_Table):
    
    def dual_simplex_loop(self,base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        
        in_index_col_have_pos = True
        no_loop_base_change = True
        no_inf_solution = True
        in_index = -1
        out_index = -1

        while(1):

            prev_in_index = in_index
            prev_out_index = out_index

            # change simplex table
            A_b, A_n, c_b, c_n, b, res = self.transform(A_b, A_n, c_b, c_n, b, res)
            if (self.debug == True):
                print("After transforming...")
                self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

            # decide end of simplex loop 
            if (self.decide_end_simplex_or_not(c_n, b)):
                if (self.debug == True): print("Judging finish simplex or not... Finally End!")
                break
            else:
                if (self.debug == True): print("Judging finish simplex or not... Keep Going!")

            # change base
            base_list, null_list, A_b, A_n, c_b, c_n, in_index_col_have_pos, in_index, out_index\
                = self.change_base(base_list, null_list, A_b, A_n, c_b, c_n, b)
            if(self.debug == True): 
                self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
            
            # detect situations like a1-->a2  a2--->a1 a1--->a2 ...
            if (prev_in_index == out_index and prev_out_index == in_index):
                no_loop_base_change = False

            # detect inf solution return type
            if (in_index_col_have_pos == False or no_loop_base_change == False):
                no_inf_solution = False
                break
        
        return base_list, null_list, A_b, A_n, c_b, c_n, b, res, no_inf_solution

    def dual_simplex_algo(self):
        print("Enter dual simplex algorithm")

        # no solution type 1 (rank error check)
        if (self.check_rank(self.A,self.c) == -1):
            return -1,0,[]
        
        # 1st stage var
        base_list = [i + (self.var_num - self.constrain_num) for i in range(self.constrain_num)]
        null_list = [i for i in range(self.var_num - self.constrain_num)]
        
        self.base_list = base_list
        self.null_list = null_list

        self.A_b, self.A_n  = self.split_matrix(self.A, self.base_list, self.null_list)
        self.c_b, self.c_n  = self.split_matrix(self.c, self.base_list, self.null_list)
        self.res = 0

        self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res, no_inf_solution = \
            self.dual_simplex_loop( \
                self.base_list, \
                self.null_list, \
                self.A_b, \
                self.A_n, \
                self.c_b, \
                self.c_n, \
                self.b, \
                self.res)

        if (no_inf_solution == False):
            return 0, 0, []

        # combine and adjutst auxiliary vars
        self.res_type, self.opt_solution, self.solution = self.handle_result(self.res_type, self.base_list, self.null_list, (self.b).reshape(-1), self.res, self.var_status)

        print(">>>add",end="\n")
        return self.res_type, self.opt_solution, self.solution


    def decide_end_simplex_or_not(self, c_n, b):
        # not 0 but -1e-5 ~ 1e-5
        # c_n might be empty, simplex should be ended in one step
        print("c_n",c_n.shape,end="\n")
        if (c_n.shape == (1,0)):
            print("c_n.shape",end="\n")
            return True
        else:
            return True if ((np.max(c_n)<=-1e-5 and np.min(b) >=1e-5)) else False

    def find_out_index(self, b, base_list):
        print(">>>b",b,end="\n")
        print(">>>max ",np.argmin(b),end = "\n")
        print(">>>null_list",base_list,end="\n")
        print(">>>null_list(max)",base_list[np.argmin(b)], end = "\n")
        return base_list[np.argmin(b)]


    def find_in_index(self,A_n, c_n, out_index, null_list,base_list):

        
        """ find min cj/|aij| (aij < 0) 
        """
        if (self.debug == True):
            print(">>>A_n",A_n,end="\n")

        out_index_pos_in_null_list = base_list.index(out_index)

        if (self.debug == True):
            print(">>>out index ",out_index_pos_in_null_list,end="\n")
        
        A_n_out_index_row = A_n[out_index_pos_in_null_list].reshape(-1,1)
        A_n_out_index_row_copy = A_n_out_index_row.copy()
        
        if (self.debug == True):
            print(">>>A_n_out ",A_n_out_index_row_copy,end = "\n")
            print(">>>c_n",c_n,end = "\n")

        c_n = c_n.reshape(-1,1)
        A_n_out_index_row_copy[A_n_out_index_row_copy == 0] = -1
        div_mat = c_n / A_n_out_index_row_copy

        if (self.debug == True):
            print(">>>div mat",div_mat,end="\n")
            print('>>>np.argmin()',np.argmin(div_mat),end="\n")

        out_index_col_have_pos = True

        if (self.debug == True):
            print(">>>div_mat value",div_mat[np.argmin(div_mat)],end="\n")

        while(np.min(div_mat)<= 1e-5):
            div_mat[np.argmin(div_mat)] = self.INF
            if(np.min(div_mat) == self.INF):
                out_index_col_have_pos = False
                break
        
        if (self.debug == True):
            print(">>>base list",base_list,end="\n")
            print('>>>null list',null_list,end="\n")
            
        return null_list[np.argmin(div_mat)], out_index_col_have_pos
    

    def change_base(self, base_list, null_list, A_b, A_n, c_b, c_n, b):
        # get out_index, in_index
        out_index = self.find_out_index(b, base_list)
        in_index, out_index_col_have_pos = self.find_in_index(A_n, c_n, out_index, null_list, base_list)

        # find index in the list
        in_index_pos_in_null_list = null_list.index(in_index)
        out_index_pos_in_base_list = base_list.index(out_index)

        # swap two columns
        A_tmp = A_n[:, in_index_pos_in_null_list].copy()
        A_n[:, in_index_pos_in_null_list] = A_b[:, out_index_pos_in_base_list]
        A_b[:, out_index_pos_in_base_list] = A_tmp

        # swap tow columns
        c_tmp = c_n[:, in_index_pos_in_null_list].copy()
        c_n[:, in_index_pos_in_null_list] = c_b[:, out_index_pos_in_base_list]
        c_b[:, out_index_pos_in_base_list] = c_tmp

        # modify two lists
        null_list[in_index_pos_in_null_list] = out_index
        base_list[out_index_pos_in_base_list] = in_index

        if (self.debug == True):
            print("Changing base... IN={}, OUT={}".format(in_index, out_index))

        return base_list, null_list, A_b, A_n, c_b, c_n, out_index_col_have_pos, in_index, out_index


if __name__ == '__main__':

    """ control program mode """
    test = False

    if (test == False):

        """ normal mode
            input required data
            output result of two algo
        """

        # 1st line of input
        # variable number and constrain number
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != 2 or int(var_constrain_str_split[0]) <=0 or int(var_constrain_str_split[1]) <= 0): 
            raise Exception('Variable and constrain number input ERROR!')
        var_num = int(var_constrain_str_split[0])
        constrain_num = int(var_constrain_str_split[1])

        # 2nd line of input
        # paramters for target function ( min -> by default)
        target_func_para_str = input()
        target_func_para_str_split = target_func_para_str.split()
        if (len(target_func_para_str_split) != var_num): 
            raise Exception('Parameters for target function input ERROR!')
        # change the (min) opt problem into (max) opt problem
        target_func_para_arr = [-int(i) for i in target_func_para_str_split]

        # init for important numpy variable
        # c must be a row vector
        c = np.array(target_func_para_arr).reshape(1,-1)
        # A,b must be empty numpy array
        A = np.empty(shape=(0, var_num))
        b = np.empty(shape=(0, 1))
        # res must be zero scalar
        res = 0

        # help the construction auxiliary variables
        constrain_equality = []
        # (constrain_number) lines lf input
        for i in range(constrain_num):
            constrain_str = input()
            constrain_str_split = constrain_str.split()
            if (len(constrain_str_split) != (var_num+2)): 
                raise Exception('input for constrain ERROR! constrain_str length = {}'.format(len(constrain_str.split())))

            constrain_para_arr = [int(i) for i in constrain_str_split]
            if (constrain_para_arr[-1] != -1 and constrain_para_arr[-1] != 0 and constrain_para_arr[-1] != 1): 
                raise Exception('input for constrain ERROR! last number in the constraint must be -1 or 1 or 0')

            # modify the constrain into standard forms ax+by <= c if it is ax+by >= c
            if (constrain_para_arr[-1] == 1):
                constrain_para_arr = [-i for i in constrain_para_arr]

            constrain_equality.append(constrain_para_arr[-1])

            row_A = np.array(constrain_para_arr[:-2]).reshape(-1,var_num)
            row_b = np.array(constrain_para_arr[-2]).reshape(-1,1)
            A = np.concatenate((A, row_A),axis=0).reshape(-1, var_num)
            b = np.concatenate((b, row_b),axis=0).reshape(-1, 1)

        # one lines for variable limitation
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != var_num):
            raise Exception('input for constraints on variables ERROR!')

        var_constrain_arr = [int(i) for i in var_constrain_str_split]
        
        dual_simplex = Dual_Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality,debug=True)
        res_type, opt_solution, solution = dual_simplex.dual_simplex_algo()
        dual_simplex.print_result()
        
        # print("std simplex algorithm")
        # res_type, opt_solution, solution = dual_simplex.std_simplex_algo()
        # dual_simplex.print_result()

    elif (test == True):

        """ test mode
            use generated random data
            use both std and self-build simplex-algo
            check error within 1e-5
        """
        