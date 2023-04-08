#define FILE_EXTENSION ".txt"
#include<fstream>
#include<string>
#include<cstring>
#include<vector>
#include<iostream>
#include <bits/stdc++.h>
#include <time.h>
using namespace std;

const int alph_size = 26;

// trie node
struct TrieNode
{
    struct TrieNode *children[alph_size];

    bool EOW;
};

 struct trie_pointer{
    TrieNode *trie = NULL;
    trie_pointer *next = NULL;
    int file_idx = -1;
    string title;
 };
struct TrieNode *getNode(void)
{
    struct TrieNode *pNode =  new TrieNode;

    pNode->EOW = false;

    for (int i = 0; i < alph_size; i++)
        pNode->children[i] = NULL;

    return pNode;
}

void Insert(struct TrieNode *root, string key)
{
    struct TrieNode *np = root;

    for (int i = 0; i < key.length(); i++)
    {
        int index;
        if(key[i]>'Z') index = key[i]-'a';
        else index = key[i] - 'A';
        if (!np->children[index])
            np->children[index] = getNode();

        np = np->children[index];
    }

    np->EOW = true;
}

bool Search(struct TrieNode *root, string key, bool is_pre)
{
    struct TrieNode *np = root;

    for (int i = 0; i < key.length(); i++)
    {
        int index;
        if(key[i]>'Z') index = key[i]-'a';
        else index = key[i] - 'A';
        if (!np->children[index])
            return false;

        np = np->children[index];
    }
    if(is_pre) return true;
    else return (np->EOW);
}

// string parser : output vector of strings (words) after parsing
vector<string> word_parse(vector<string> tmp_string){
	vector<string> parse_string;
	for(auto& word : tmp_string){
		string new_str;
    	for(auto &ch : word){
			if(isalpha(ch))
				new_str.push_back(ch);
		}
		parse_string.emplace_back(new_str);
	}
	return parse_string;
}

vector<string> split(const string& str, const string& delim) { //split for string
	vector<string> res;
	if("" == str) return res;
	//先將要切割的字串從string型別轉換為char*型別
	char * strs = new char[str.length() + 1] ; //不要忘了
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while(p) {
		string s = p; //分割得到的字串轉換為string型別
		res.push_back(s); //存入結果陣列
		p = strtok(NULL, d);
	}

	return res;
}

void file_to_sufftrie(fstream& fi, trie_pointer* p){
    TrieNode* root = getNode();
    p->trie = root;
    string tmp;
    vector<string> tmp_string;
    bool is_title = true;
    while(getline(fi, tmp)){
        if(is_title) p->title =tmp;
		tmp_string = split(tmp, " ");

		vector<string> content = word_parse(tmp_string);

		for(auto &word : content){
			int l = word.length();
			for(int i=l-1, cnt=1;i>=0;i--, cnt++){
                string ins = word.substr(i, cnt);
                Insert(root, ins);
			}
		}
		is_title = false;
	}
}

void file_to_trie(fstream& fi, trie_pointer* p){
    TrieNode* root = getNode();
    p->trie = root;
    string tmp;
    vector<string> tmp_string;
    bool is_title = true;
    while(getline(fi, tmp)){
        if(is_title) p->title =tmp;
		tmp_string = split(tmp, " ");

		vector<string> content = word_parse(tmp_string);

		for(auto &word : content){
			Insert(root, word);
		}
		is_title = false;
	}
}

int main(int argc, char *argv[])
{
    //clock_t start,End;
    //start=clock();

    // INPUT :
	// 1. data directory in data folder
	// 2. number of txt files
	// 3. output route

    string data_dir = argv[1] + string("/"); //data/
	string query = string(argv[2]);  //query.txt
	string output = string(argv[3]); //output.txt

	// Read File & Parser Example


	string file, title_name, tmp;
	fstream fi;

	//from directory, build a trie for every file, use linked list(starts from lhead->next) to save every trie
    trie_pointer *lhead = new trie_pointer();
    trie_pointer *suffhead = new trie_pointer();
	int idx=0;
	string temp_path = data_dir + to_string(idx) + ".txt";
	auto path = temp_path.c_str();
    fi.open(path, ios::in);
    trie_pointer *np, *p;
    trie_pointer *snp, *sp;
    p = lhead;
    sp  = suffhead;
	while(fi){
//cout<<"building trie for file "<<idx<<".txt\n";
        np = new trie_pointer();
        np->file_idx = idx;
        file_to_trie(fi, np);
        //build suffix trie
        fi.clear();
        fi.seekg(0);
        snp = new trie_pointer();
        snp -> file_idx = idx;
        file_to_sufftrie(fi, snp);
//cout<<"done building trie for file "<<idx<<".txt\n";
        fi.close();
        p->next = np;
        p = np;
        sp ->next = snp;
        sp = snp;
        idx++;
        temp_path = data_dir + to_string(idx) + ".txt";
        path = temp_path.c_str();
        fi.open(path, ios::in);
	}
//cout<<"finished building trie for all files\n";

	//fi.open("data/0.txt", ios::in);
    temp_path = query;
    path = temp_path.c_str();
    fi.open(path, ios::in);

    fstream fout;
    temp_path = output;
    path = temp_path.c_str();
    fout.open(path, ios::out);

    vector<string> tmp_words;
    int qcount=1;
    while(getline(fi, tmp)){ //every  query
//cout<<"in query\n\n";
		tmp_words = split(tmp, " "); //get search words and "/", "+"
        set<pair<int, string>> file_set;
        string op = "";
		for(auto &w : tmp_words){//for every search word
            set<pair<int, string>> nset;
            if(w=="/" || w=="+"){
                op = w;
            }else{
                string key;
                bool is_prefix = false;
                if(w[0]=='*'){// search for suffix
                    key = w.substr(1, w.length()-2);
                    trie_pointer* np = suffhead->next;
                    while(np){ //search through every file
                        if(Search(np->trie, key, is_prefix)){
                            nset.insert(make_pair(np->file_idx, np->title));
                        }
                        np = np->next;
                    }

                }
                else{
                    if(w[0]=='\"'){//search for whole word
                        key = w.substr(1, w.length()-2);
                        is_prefix = false;
                    }
                    else{ //search for prefix
                        is_prefix = true;
                        key = w;
                    }
                    trie_pointer* np = lhead->next;
                    while(np){ //search through every file
                        if(Search(np->trie, key, is_prefix)){
                            nset.insert(make_pair(np->file_idx, np->title));
                        }
                        np = np->next;
                    }
                }
//cout<<"done searching for: "<<key<<endl;
                if(op!=""){
                    if(op=="+"){ //intersect two sets
                        vector<pair<int, string>> common_data;
                        set_intersection(file_set.begin(),file_set.end(),nset.begin(),nset.end(), std::back_inserter(common_data));
                        file_set.clear();
                        for(auto i:common_data){
                            file_set.insert(i);
                        }
//cout<<"done sets intersect\nthe set is:";
 //for(auto it: file_set){
 //           cout<<it.first<<" ";
//}
//cout<<endl;
                    }
                    else if(op=="/"){ //union two sets
                        file_set.insert(nset.begin(), nset.end());
//cout<<"done sets union\nthe set is:";
 //for(auto it: file_set){
  //          cout<<it.first<<" ";
//}
//cout<<endl;
                    }
                    op = "";
                }
                else{
                    file_set.insert(nset.begin(), nset.end());
//cout<<"done initializing set\n";
                }
            }

		}
//cout<<"done searching\n";
        if(file_set.empty()) fout<<"Not Found!"<<endl;
        else{
            for(auto it: file_set){
            fout<<it.second<<endl;
            }
        }
//cout<<"done writing titles to output file for query "<<qcount<<endl;
//qcount++;
	}
	fi.close();
	fout.close();
    //End=clock();
  //  cout << (double)((double)(End-start)/CLOCKS_PER_SEC)<<endl;
}


// 1. UPPERCASE CHARACTER & LOWERCASE CHARACTER ARE SEEN AS SAME.
// 2. FOR SPECIAL CHARACTER OR DIGITS IN CONTENT OR TITLE -> PLEASE JUST IGNORE, YOU WONT NEED TO CONSIDER IT.
//    EG : "AB?AB" WILL BE SEEN AS "ABAB", "I AM SO SURPRISE!" WILL BE SEEN AS WORD ARRAY AS ["I", "AM", "SO", "SURPRISE"].
// 3. THE OPERATOR IN "QUERY.TXT" IS LEFT ASSOCIATIVE
//    EG : A + B / C == (A + B) / C

//

//////////////////////////////////////////////////////////
