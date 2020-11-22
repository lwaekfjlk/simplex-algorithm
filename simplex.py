import numpy as np
import random



class Simplex_Table(object):
    def __init__(self,ori_c,ori_A,ori_b,ori_res,var_limit_arr,cons_type_arr):  
        # constant variable used by var_status
        # NORMAL state means normal variable x
        # NINF   state means the x actually is replaced by (-x)
        # POS    state means the variable is a pair with another variable x+ and x- separated from x in R
        self.NINF = -1000000
        self.INF  =  1000000
        self.NORMAL = -1
        self.ori_constrain_num = ori_A.shape[0]
        self.ori_var_num = ori_A.shape[1]
        self.var_status = np.full(self.ori_var_num,self.NORMAL)
        
        # change variable based on variable limitation
        for i in range(len(var_limit_arr)):
            # if a variable has <= 0 limitation
            if (var_limit_arr[i] == -1):
                ori_A[:,i] = -ori_A[:,i]
                ori_c[:,i] = -ori_c[:,i]
                self.var_status[i] = self.NINF
            # if a variable has no limitation
            if (var_limit_arr[i] == 0):
                col_A = np.array(-ori_A[:,i]).reshape(-1,1)
                col_c = np.array(-ori_c[:,i]).reshape(-1,1)
                ori_A =  np.concatenate((ori_A, col_A),axis=1).reshape(constrain_num,-1)
                ori_c =  np.concatenate((ori_c, col_c),axis=1).reshape(1,-1)
                self.var_status[i] = i
                self.var_status = np.append(self.var_status, i)
        
        # add AUX variable based on constrain type 
        for i in range(len(cons_type_arr)):
            if (cons_type_arr[i] == -1):
                col_A = np.zeros(shape=(self.ori_constrain_num,1))
                col_A[i][0] = 1
                ori_A = np.concatenate((ori_A,col_A), axis=1).reshape(constrain_num,-1)
                col_c = np.zeros(shape=(1,1))    
                ori_c = np.concatenate((ori_c,col_c), axis=1).reshape(constrain_num,-1)

        self.c = ori_c
        self.A = ori_A
        self.b = ori_b
        self.res = ori_res
        self.res_type = 1
        self.constrain_num = (self.A).shape[0]
        self.var_num = (self.A).shape[1]

        print("modified target  matrix for c = \n{}".format(self.c))
        print("modified constrain matrix for A = \n{}".format(self.A))
        print("modified constrain matrix for b = \n{}".format(self.b))
        print("recorded state for variables = \n{}".format(self.var_status))

    def print_simplex_table(self, base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("======",end='')
        print("+",end='')
        print("\n",end='')

        # base number
        print("|",end="")
        for i in range(len(base_list)):
            print("%5d"%base_list[i],end=" ")
        for i in range(len(null_list)):
            print("%5d"%null_list[i],end=" ")
        print("      |\n",end="")

        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("======",end='')
        print("+",end='')
        print("\n",end='')
        
        # c arr line
        print("|",end="")
        for i in range(c_b.shape[1]):
            print("%5.2f"%c_b[0][i],end=" ")
        for i in range(c_n.shape[1]):
            print("%5.2f"%c_n[0][i],end=" ")
        print("|%5.2f"%res,end="")
        print("|\n",end="")

        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("------",end='')
        print("+",end='')
        print("\n",end='')

        # A | b  matrix
        for i in range(self.constrain_num):
            print("|",end="")
            for j in range(A_b.shape[1]):
                print("%5.2f"%A_b[i][j],end=" ")
            for j in range(A_n.shape[1]):
                print("%5.2f"%A_n[i][j],end=" ")
            print("|%5.2f"%b[i],end="")
            print("|\n",end="")
        
        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("======",end='')
        print("+",end='')
        print("\n",end='')        



    def split_matrix(self,mat,lista, listb):
        return mat[:,lista], mat[:,listb]

    def inv_matrix(self,mat):
        return np.linalg.inv(mat)
    
    def transform(self, A_b, A_n, c_b, c_n, b, res):
        res = res - float(np.dot(np.dot(c_b, self.inv_matrix(A_b)) , b))
        c_n = c_n - np.dot(np.dot(c_b, self.inv_matrix(A_b)), A_n)
        A_n = np.dot(self.inv_matrix(A_b), A_n)
        b   = np.dot(self.inv_matrix(A_b), b)
        A_b = np.eye(A_b.shape[0])
        c_b = np.zeros(shape=(c_b.shape))
        return A_b, A_n, c_b, c_n, b, res
    
    def find_inf_solution(self, A, c, row_num):
        # A_tmp = np.concatenate((A,c), axis=0)
        # incorrect!!!!!!!!!!!!!!!
        # return True if(np.linalg.matrix_rank(A_tmp) == row_num) else False
        return False

    def find_A_not_full_row_rank(self, A, row_num):
        return True if (np.linalg.matrix_rank(A) != row_num) else False

    def have_special_situation(self, A, c):
        find_inf_solution_res  =  self.find_inf_solution(A, c, A.shape[0])
        find_A_not_full_row_rank_res = self.find_A_not_full_row_rank(A, A.shape[0])
        if (find_inf_solution_res == True):
            return 0
        if (find_A_not_full_row_rank_res == True):
            return -1
        return 1

    def decide_end_simplex_or_not(self, c_n):
        return True if (np.max(c_n)<=0) else False

    def find_in_index(self, c_n, null_list):
        return null_list[np.argmax(c_n)]

    def find_out_index(self, A_n, b, in_index, null_list,base_list):
        in_index_pos_in_null_list = null_list.index(in_index)
        A_n_in_index_col = A_n[:,in_index_pos_in_null_list].reshape(-1,1)
        A_n_in_index_col_copy = A_n_in_index_col.copy()

        # since we might come across divide zero error, we change all 0 in the np array into -1
        # definitely we would not choose the 0 and the -1, so the effect is the same
        A_n_in_index_col_copy[A_n_in_index_col_copy == 0] = -1
        div_mat = b / A_n_in_index_col_copy

        # if the minimal elememnt in the div_mat has a negative corresponding A_n_in_index_col
        while(A_n_in_index_col_copy[np.argmin(div_mat)] <= 0):
            div_mat[np.argmin(div_mat),:] = np.array(self.INF).reshape(1,-1)
        return base_list[np.argmin(div_mat)]

    def change_base(self, base_list, null_list, A_b, A_n, c_b, c_n, b):
        in_index  = self.find_in_index(c_n, null_list)
        out_index = self.find_out_index(A_n, b, in_index, null_list, base_list)
        
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

        return base_list, null_list, A_b, A_n, c_b, c_n

    def simplex_loop(self,base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        while (1):
            A_b, A_n, c_b, c_n, b, res = self.transform(A_b, A_n, c_b, c_n, b, res)
            print("After transforming...")
            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

            print("Judging finished simplex or not...")
            end_of_simplex = self.decide_end_simplex_or_not(c_n)
            if (end_of_simplex == True): 
                    print("That is it, finally end!")
                    break
            else:
                print("We need to keep going!")

            base_list, null_list, A_b, A_n, c_b, c_n = self.change_base(base_list, null_list, A_b, A_n, c_b, c_n, b)
            print("After changing base...")
            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
        
        return base_list, null_list, A_b, A_n, c_b, c_n, b, res


    def simplex_algo(self):
        # construct the initial simplex table
        # error check
        self.res_type = self.have_special_situation(self.A, self.c)
        if (self.res_type != 1):
            return
        
        # 1st stage for 2 stage simplex method
        base_list = [i+self.var_num for i in range(self.constrain_num)]
        null_list = [i for i in range(self.var_num)]
        A_n = self.A.copy()
        A_b = np.eye(self.constrain_num)
        c_n = np.zeros(shape=(1,self.var_num))
        c_b = np.full((1,self.constrain_num),-1)
        b   = self.b.copy()   
        res = 0

        print("\nBegin 1nd Stage for simplex method!")
        base_list, null_list, A_b, A_n, c_b, c_n, b, res = self.simplex_loop(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

        print("\nInitial point FOUND !")
        self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

        # in the 1 stage of simplex method, we can find situation that can not find solution for simplex method
        if(res != 0):
            self.res_type = -1
            return
        
        self.null_list = [i for i in null_list if i < self.var_num]
        self.base_list = [i for i in base_list if i < self.var_num]
        if (len(self.base_list) < self.constrain_num):
            self.base_list += self.null_list[-(self.constrain_num - len(self.base_list)):]
            self.null_list  = self.null_list[:-(self.constrain_num - len(self.base_list))]
        self.A_b, self.A_n  = self.split_matrix(self.A, self.base_list, self.null_list)
        self.c_b, self.c_n  = self.split_matrix(self.c, self.base_list, self.null_list)
        
        print("\nBegin 2nd Stage for simplex method!")
        self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)

        if (len(self.null_list) > 0):
            self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res = self.simplex_loop(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
        else:
            # I once come into a circumstances that null_list = [] and the inv of A_b is quite  strange and Ax = b cannnot be solved by A-1*b
            self.res = self.res - float(np.dot(np.dot(self.c_b, self.inv_matrix(self.A_b)) , self.b))
            self.b = np.dot(self.inv_matrix(self.A_b), self.b)

        print("\nComplete, get final base...")
        self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
        
        self.handle_result(self.res_type, self.base_list, self.null_list, (self.b).reshape(-1), self.res, self.var_status)

        return

    def handle_result(self, res_type, base_list, null_list, base_solution, opt_solution, var_status):
        # the first two lines for answers

        print(res_type)
        print(opt_solution)

        # construct the solution list 
        base_cnt = 0
        null_cnt = 0
        solution = [0 for i in range(len(base_list+null_list))]

        for i in range(len(base_list)):
            solution[base_list[i]] = base_solution[i]

        # change solution list based on var_status
        for i in range(len(solution)):
            if (var_status[i] == self.NINF):
                solution[i] = -solution[i]
            elif (var_status[i] == self.NORMAL):
                continue
            elif (var_status[i] != i):
                solution[var_status[i]] += solution[i]
        
        #  print the final soltion list
        for i in range(self.var_num-1):
            print(solution[i],end=" ")
        print(solution[self.var_num-1])
        return


if __name__ == '__main__':

    # 1st line of input
    # variable number and constrain number
    var_constrain_str = input()
    var_constrain_str_split = var_constrain_str.split()
    if (len(var_constrain_str_split) != 2): 
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

    '''
    print("variable_number = {}".format(var_num))
    print("constrain_number = {}".format(constrain_num))
    print("standard target matrix for c = {}".format(c))
    print("standard constrain matrix for A = {}".format(A))
    print("standard constrain matrix for b = {}".format(b))
    '''

    simplex = Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality)
    simplex.simplex_algo()


