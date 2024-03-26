
import requests

def main(): # Main Function.
    while True:
        # api_token = 'eb2ba0e78ef2daab5fe9f55ff273651b'
        api_url_base = 'http://api.openweathermap.org/data/2.5/weather?APPID=eb2ba0e78ef2daab5fe9f55ff273651b'

        prompt = 'Hello what is you name?\n' # Welcome message.
        name = input(prompt)
        print('Welcome', name + '.')

        prompt = 'What is the 5 digit zipcode you wish to use?\n' # Obtain the zipcode.
        chosen_zip_code = input(prompt)
        print ('Thank you, the zipcode you have chosen is:', chosen_zip_code)
        use_api(chosen_zip_code, api_url_base)

        prompt = 'Would you like to restart the program Yes or No?\n'
        answer = input(prompt)

        if answer =='No': # Stops the program.
            break
        elif answer =='Yes': # Restarts the program.
            continue
        else: # Ends the program.
            print('Invalid input')
            print('Ending Program.')
        break


def pretty_print(json_zip): # Print results in readable format.
    my_list = list ()
    for key, val in list(json_zip.items()):
        my_list.append((val, key))

    for key, val in my_list[:]:
        print('{:15s}{:15s}'.format(str(val), str(key)))


def use_api(zip_code, api_url_base):
    url_zip = api_url_base + '&zip=' + zip_code #Add the zipcode to the request.
    zip_response = requests.get(url_zip)
    json_zip = zip_response.json()
    try:
        if zip_response.status_code == 200: # Validates if request was successful.
            print ('Request was successful.')
            print('----------------------')
            pretty_print(json_zip)
        else: # Validates if request was not successful.
            print('Request was not successful.')
            return None
    except:
        print('An exception occurred.')

main()
