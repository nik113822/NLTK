import pytest
from src.nba_anal.anal2 import clean_text,process_full

  
def test_clean_text_first_tweet():
    text = "Hello!!! @NBA https://nba.com Lakers WIN 😂"
    assert clean_text(text) == "hello lakers win"

def test_clean_text_second_tweet():
    #5hgrammh dataset
    k = "Check out Allen Iverson basketball jersey Sixers size XL #Champion #Philadelphia76ers https://t.co/0miMCzErEW via @eBay"
    assert clean_text(k) == "check out allen iverson basketball jersey sixers size xl champion philadelphia76ers via"

def test_clean_text_3rd_tweet():
    #45h grammh
    r = "The Detroit Pistons Offer J.Cole His First NBA Tryout #SourceSports #TheSource #JCole #NBA #DetroitPistons https://t.co/S9taBrQWh0"
    assert clean_text(r) == "detroit pistons offer his first nba tryout sourcesports thesource jcole nba detroitpistons"

def test_clean_text_4th_tweet():
    #87h grammh
    s = " Nate McMillan agree to contract extension #IndianaPacers #NateMcMillan https://t.co/mwzzItEW6D" 
    assert clean_text(s) == "nate mcmillan agree to contract extension indianapacers natemcmillan"


def test_process_full_5th_tweet():
    #29501h grammh
    w = "bs feeble grind #skateporamor❤️#charlottehornets #raiders #nikesb https://t.co/xFQEfXjGdC,1296618971333120000" 
    assert process_full(w) == "b feeble grind skateporamor charlottehornets raider nikesb"

def test_process_full_6th_tweet():
    #34678h grammh
    d = "Check out QUINCY ACY 2012-13 Hoops # 293 RC  #Hoops #TorontoRaptors https://t.co/cgPnEdpUV7 via @eBay" 
    assert process_full(d) == "check quincy acy 2012 13 hoop 293 rc hoop torontoraptors via"

def test_process_full_7th_tweet():
    #44333h grammh
    z ="Why did Lopez gently caress Adebayo's head like that?? #NBA #MilwaukeeBucks #MiamiHeat #MIAvsMIL https://t.co/DkDjpaNNRt" 
    assert process_full(z) == "lopez gently caress adebayo head like nba milwaukeebucks miamiheat miavsmil"


def test_clean_text_only_symbols_emojis():
    assert clean_text("!!! 😂 ❤️ :)") == ""

def test_clean_text_mentions():
    assert clean_text("@") == ""
    




