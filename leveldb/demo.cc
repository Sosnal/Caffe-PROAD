#include <iostream>

#include <string>

#include <leveldb/db.h>

using namespace std;

int main()

{
   
    leveldb::DB *db;

    leveldb::Options options;

    options.create_if_missing = true;

    leveldb::Status status = leveldb::DB::Open(options, "/tmp/testdb", &db);

    cout <<"status:" << status.ok() << endl;

    if(!status.ok()){

        return 1;

    }

    string key = "apple";

    string value = "A";

    string get;


    leveldb::Status s = db->Put(leveldb::WriteOptions(), key, value);


    if (s.ok())

        s = db->Get(leveldb::ReadOptions(), key, &get);

    if (s.ok())

        cout << "key:" << key << "\nget value:" << get << std::endl;

    else

        cout << "error occurred!" << endl;


    delete db;


    return 0;

}
